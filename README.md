# 量化交易智能助手

这是一个基于 smolagents 和 Gradio 构建的量化交易智能助手系统，集成了 Tushare 数据接口和回测功能。

## 功能特点

- 支持通过 Tushare 获取股票数据
- 集成 backtesting.py 进行策略回测
- 提供友好的 Web 界面
- 支持多种量化分析工具
- 支持 OpenTelemetry 监控
- 支持通过.env文件进行配置

## 快速开始

1. 克隆项目到本地
2. 复制`.env.example`文件为`.env`并填写配置：
   ```
   # 模型配置
   MODEL_ID=qwen-max
   API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
   API_KEY=your_api_key_here

   # Tushare配置
   TUSHARE_TOKEN=your_tushare_token_here

   # 启用的工具（用逗号分隔）
   ENABLED_TOOLS=get_tushare_daily_bar,get_stock_basic,backtesting_py_tool

   # 文件上传配置（可选）
   FILE_UPLOAD_FOLDER=uploads
   ```
3. 安装依赖：`pip install -r requirements.txt`
4. 运行程序：`python QuantAgent.py`

## 配置说明

系统支持通过.env文件进行配置，主要配置项包括：

- `MODEL_ID`: 使用的模型ID（如 qwen-max）
- `API_BASE`: API基础URL
- `API_KEY`: API密钥
- `TUSHARE_TOKEN`: Tushare的访问令牌
- `ENABLED_TOOLS`: 启用的工具列表，用逗号分隔
- `FILE_UPLOAD_FOLDER`: 文件上传目录（可选）

## 可用工具

- `get_tushare_daily_bar`: 获取股票日线数据
- `get_stock_basic`: 获取股票基本信息
- `backtesting_py_tool`: 执行策略回测
- `DuckDuckGoSearchTool`: 网络搜索（可选）
- `VisitWebpageTool`: 访问网页（可选）

## 监控

系统集成了 OpenTelemetry 监控，可以通过 http://localhost:6006 访问监控界面。

## 注意事项

- 请确保在使用前正确配置.env文件
- 敏感信息（如API密钥）应该只保存在.env文件中，不要提交到版本控制系统
- 回测时请注意数据的时间范围和质量
- 建议在虚拟环境中运行项目

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。
