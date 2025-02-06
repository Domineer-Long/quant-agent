# smolagents_quant_analysis.py
import os
import pandas as pd
from smolagents import (
    CodeAgent,
    OpenAIServerModel,
    ToolCallingAgent,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    ManagedAgent,
    tool,
)
import gradio as gr
import tushare as ts
import re
import mimetypes
import os
import shutil
from typing import Optional

from smolagents.agent_types import (
    AgentAudio,
    AgentImage,
    AgentText,
    handle_agent_output_types,
)
from smolagents.agents import ActionStep, MultiStepAgent
from smolagents.memory import MemoryStep
from smolagents.utils import _is_package_available

# import matplotlib.pyplot as plt

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://localhost:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)


def pull_messages_from_step(
    step_log: MemoryStep,
):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    import gradio as gr

    if isinstance(step_log, ActionStep):
        # Output the step number
        step_number = (
            f"Step {step_log.step_number}" if step_log.step_number is not None else ""
        )
        yield gr.ChatMessage(role="assistant", content=f"**{step_number}**")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(
                r"```\s*<end_code>", "```", model_output
            )  # handles ```<end_code>
            model_output = re.sub(
                r"<end_code>\s*```", "```", model_output
            )  # handles <end_code>```
            model_output = re.sub(
                r"```\s*\n\s*<end_code>", "```", model_output
            )  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield gr.ChatMessage(role="assistant", content=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(
                    r"```.*?\n", "", content
                )  # Remove existing code blocks
                content = re.sub(
                    r"\s*<end_code>\s*", "", content
                )  # Remove end_code tags
                content = content.strip()
                if not content.startswith("```python"):
                    content = f"```python\n{content}\n```"

            parent_message_tool = gr.ChatMessage(
                role="assistant",
                content=content,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "pending",
                },
            )
            yield parent_message_tool

            # Nesting execution logs under the tool call if they exist
            if hasattr(step_log, "observations") and (
                step_log.observations is not None and step_log.observations.strip()
            ):  # Only yield execution logs if there's actual content
                log_content = step_log.observations.strip()
                if log_content:
                    log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"{log_content}",
                        metadata={
                            "title": "📝 Execution Logs",
                            "parent_id": parent_id,
                            "status": "done",
                        },
                    )

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                    metadata={
                        "title": "💥 Error",
                        "parent_id": parent_id,
                        "status": "done",
                    },
                )

            # Update parent message metadata to done status without yielding a new message
            parent_message_tool.metadata["status"] = "done"

        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error"},
            )

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(
            step_log, "output_token_count"
        ):
            token_str = f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            step_footnote += token_str
        if hasattr(step_log, "duration"):
            step_duration = (
                f" | Duration: {round(float(step_log.duration), 2)}"
                if step_log.duration
                else None
            )
            step_footnote += step_duration
        step_footnote = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
        yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    if not _is_package_available("gradio"):
        raise ModuleNotFoundError(
            "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
        )
    import gradio as gr

    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(
        task, stream=True, reset=reset_agent_memory, additional_args=additional_args
    ):
        # Track tokens if model provides them
        if hasattr(agent.model, "last_input_token_count"):
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(
            step_log,
        ):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(
            role="assistant", content=f"**Final answer:** {str(final_answer)}"
        )


@tool
def get_tushare_daily_bar(ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取股票的日线行情数据。

    Args:
        ts_code: 股票代码（如 '000001.SZ'）
        start_date: 开始日期，格式为 'YYYYMMDD'
        end_date: 结束日期，格式为 'YYYYMMDD'

    Returns:
        名称	类型	描述
        ts_code	str	股票代码
        trade_date	str	交易日期
        open	float	开盘价
        high	float	最高价
        low	float	最低价
        close	float	收盘价
        pre_close	float	昨收价【除权价，前复权】
        change	float	涨跌额
        pct_chg	float	涨跌幅 【基于除权后的昨收计算的涨跌幅：（今收-除权昨收）/除权昨收 】
        vol	float	成交量 （手）
        amount	float	成交额 （千元）
    """
    try:
        pro = ts.pro_api()

        # 获取日线数据
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return "未找到数据"

        # 按日期降序排序
        df = df.sort_values("trade_date", ascending=False)

        # 将数据转换为字符串格式返回
        return df

    except Exception as e:
        return f"获取数据时发生错误: {str(e)}"


@tool
def get_stock_basic(
    name: str = "", market: str = "", list_status: str = "L"
) -> pd.DataFrame:
    """
    获取股票基础信息数据，包括股票代码、名称、上市日期、所属行业等。

    Args:
        name: 股票名称，可选参数
        market: 市场类别，可选参数（主板/创业板/科创板/CDR/北交所）
        list_status: 上市状态，默认为'L'。 L-上市 D-退市 P-暂停上市

    Returns:
        名称	类型	默认显示	描述
        ts_code	str	Y	TS代码
        symbol	str	Y	股票代码
        name	str	Y	股票名称
        area	str	Y	地域
        industry	str	Y	所属行业
        fullname	str	N	股票全称
        enname	str	N	英文全称
        cnspell	str	Y	拼音缩写
        market	str	Y	市场类型（主板/创业板/科创板/CDR）
        exchange	str	N	交易所代码
        curr_type	str	N	交易货币
        list_status	str	N	上市状态 L上市 D退市 P暂停上市
        list_date	str	Y	上市日期
        delist_date	str	N	退市日期
        is_hs	str	N	是否沪深港通标的，N否 H沪股通 S深股通
        act_name	str	Y	实控人名称
        act_ent_type	str	Y	实控人企业性质
    """
    try:
        pro = ts.pro_api()
        # 获取股票基础信息
        df = pro.stock_basic(
            name=name,
            market=market,
            list_status=list_status,
            fields="ts_code,symbol,name,area,industry,market,list_date,act_name",
        )

        if df is None or df.empty:
            return "未找到符合条件的股票信息"

        # 将数据转换为字符串格式返回
        return df

    except Exception as e:
        return f"获取数据时发生错误: {str(e)}"


from backtesting import Backtest, Strategy


@tool
def backtesting_py_tool(stock_data: pd.DataFrame, strategy: Strategy) -> pd.Series:
    """
    回测backtesting的策略，返回回测结果stats, 其中stats["_trades"]是所有的交易详情

    Args:
        stock_data: 原始股票行情数据，OHLCV都是小写字母
        strategy: 传入实现好的backtesting.Strategy，用到的OHLCV都是首字母大写

    Returns:
        回测结果
    """
    # 创建列名映射字典
    column_mapping = {
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "vol": "Volume",  # tushare的成交量列名是'vol'
    }

    # 重命名列
    stock_data = stock_data.rename(columns=column_mapping)

    # 设置日期索引
    stock_data["trade_date"] = pd.to_datetime(stock_data["trade_date"], format="%Y%m%d")
    stock_data.set_index("trade_date", inplace=True)

    # 初始化回测
    bt = Backtest(stock_data, strategy, cash=100000, commission=0.002)
    stats = bt.run()
    return stats


class QuantGradioUI:

    def __init__(
        self,
        model_id: str,
        api_base: str,
        api_key: str,
        tushare_token: str,
        enabled_tools: list,
        file_upload_folder: str | None = None,
    ):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )

        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        self.tushare_token = tushare_token
        self.enabled_tools = enabled_tools
        self.agent = None
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        print(self.initialize_agent())
        import gradio as gr

        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox("No file uploaded", visible=True), file_uploads_log

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return gr.Textbox(f"Error: {e}", visible=True), file_uploads_log

        if mime_type not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(
            self.file_upload_folder, os.path.basename(sanitized_name)
        )
        shutil.copy(file.name, file_path)

        return gr.Textbox(
            f"File uploaded: {file_path}", visible=True
        ), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
        )

    def launch(self, **kwargs):
        import gradio as gr

        with gr.Blocks(fill_height=True) as demo:
            # 配置部分
            with gr.Row():
                with gr.Column():
                    model_id = gr.Textbox(
                        label="Model ID",
                        value=self.model_id,
                        placeholder="输入模型ID，如qwen-max",
                    )
                    api_base = gr.Textbox(
                        label="API Base",
                        value=self.api_base,
                        placeholder="输入API基础URL",
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        value=self.api_key,
                        placeholder="输入API密钥",
                        type="password",
                    )
                    tushare_token = gr.Textbox(
                        label="Tushare Token",
                        value=self.tushare_token,
                        placeholder="输入Tushare Token",
                        type="password",
                    )
                with gr.Column():
                    tools_checkboxes = gr.CheckboxGroup(
                        choices=[
                            "get_tushare_daily_bar",
                            "get_stock_basic",
                            "backtesting_py_tool",
                            "DuckDuckGoSearchTool",
                            "VisitWebpageTool",
                        ],
                        value=self.enabled_tools,
                        label="启用的工具",
                    )
                    new_session_btn = gr.Button("创建新会话")

            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
                ),
                resizeable=True,
                scale=1,
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file")
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, visible=False
                )
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

            def update_config(
                model_id_val, api_base_val, api_key_val, tushare_token_val, tools_val
            ):
                self.model_id = model_id_val
                self.api_base = api_base_val
                self.api_key = api_key_val
                self.tushare_token = tushare_token_val
                self.enabled_tools = tools_val
                self.agent = None  # 重置agent，等待下次交互时重新初始化
                return []  # 返回空的消息列表来清空聊天记录

            # 事件绑定
            new_session_btn.click(
                fn=update_config,
                inputs=[model_id, api_base, api_key, tushare_token, tools_checkboxes],
                outputs=[chatbot],
            )

        demo.launch(debug=True, share=False, server_name="0.0.0.0", **kwargs)

    def initialize_agent(self):
        if not self.agent:  # 如果agent还没有初始化
            try:
                # 初始化model
                model = OpenAIServerModel(
                    model_id=self.model_id,
                    api_base=self.api_base,
                    api_key=self.api_key,
                )

                # 如果tushare还未初始化，尝试初始化
                try:
                    ts.set_token(self.tushare_token)
                except Exception as e:
                    return f"Tushare初始化失败: {str(e)}"

                # 准备tools列表
                tools = []
                if "get_tushare_daily_bar" in self.enabled_tools:
                    tools.append(get_tushare_daily_bar)
                if "get_stock_basic" in self.enabled_tools:
                    tools.append(get_stock_basic)
                if "backtesting_py_tool" in self.enabled_tools:
                    tools.append(backtesting_py_tool)

                # 如果启用了web搜索工具，创建web agent
                managed_agents = []
                if any(
                    tool in self.enabled_tools
                    for tool in ["DuckDuckGoSearchTool", "VisitWebpageTool"]
                ):
                    web_tools = []
                    if "DuckDuckGoSearchTool" in self.enabled_tools:
                        web_tools.append(DuckDuckGoSearchTool())
                    if "VisitWebpageTool" in self.enabled_tools:
                        web_tools.append(VisitWebpageTool())

                    web_agent = ToolCallingAgent(
                        tools=web_tools,
                        model=model,
                        max_steps=10,
                    )
                    managed_agents.append(
                        ManagedAgent(
                            agent=web_agent,
                            name="search",
                            description="Runs web searches for you. Give it your query as an argument.",
                        )
                    )

                # 创建CodeAgent
                self.agent = CodeAgent(
                    tools=tools,
                    model=model,
                    managed_agents=managed_agents if managed_agents else None,
                    max_steps=6,
                    verbosity_level=1,
                    additional_authorized_imports=[
                        "io",
                        "csv",
                        "tushare",
                        "numpy",
                        "pandas",
                        "time",
                        "talib",
                        "backtesting",
                    ],
                )
                return "Agent初始化成功"
            except Exception as e:
                return f"初始化失败: {str(e)}"
        return "Agent已经初始化"


# 替换原来的GradioUI调用
if __name__ == "__main__":
    ui = QuantGradioUI(
        model_id="qwen-max",
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-c99d24c4c9b8430a9cf75348a502bea5",  # 在这里填入你的API key
        tushare_token="ae45ea2d09d097f2045527c30298c3ec1ea3ae62ae5e57038dfd1d9b",  # 在这里填入你的Tushare token
        enabled_tools=[
            "get_tushare_daily_bar",
            "get_stock_basic",
            "backtesting_py_tool",
        ],
    )
    ui.launch()
