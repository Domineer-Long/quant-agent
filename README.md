# 量化交易智能助手

这是一个基于 smolagents 和 Gradio 构建的量化交易智能助手系统，集成了 Tushare 数据接口和回测功能。

## 功能特点

- 支持通过 Tushare 获取股票数据
- 集成 backtesting.py 进行策略回测
- 提供友好的 Web 界面
- 支持多种量化分析工具
- 支持 OpenTelemetry 监控

## 快速开始


## 配置说明

在运行程序前，需要设置以下配置：

- `model_id`: 使用的模型ID（如 qwen-max）
- `api_base`: API基础URL
- `api_key`: API密钥
- `tushare_token`: Tushare的访问令牌

## 可用工具

- `get_tushare_daily_bar`: 获取股票日线数据
- `get_stock_basic`: 获取股票基本信息
- `backtesting_py_tool`: 执行策略回测
- `DuckDuckGoSearchTool`: 网络搜索（可选）
- `VisitWebpageTool`: 访问网页（可选）

## 监控

系统集成了 OpenTelemetry 监控，可以通过 http://localhost:6006 访问监控界面。

## 注意事项

- 请确保在使用前正确配置 API 密钥和 Tushare token
- 回测时请注意数据的时间范围和质量
- 建议在虚拟环境中运行项目

## 许可证

MIT License

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。
