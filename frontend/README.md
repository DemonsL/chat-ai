# Chat AI 前端页面

这是一个仿照 ChatGPT 风格设计的前端页面，用于与后端 API 进行交互实现智能对话功能。

## 功能特性

### ✨ 主要功能
- **用户认证**: 支持用户注册和登录
- **对话管理**: 创建、删除、重命名对话
- **实时聊天**: 支持流式响应的实时对话
- **模型管理**: 管理员可以添加、编辑、查看AI模型配置
- **界面风格**: 仿 ChatGPT 的深色主题设计
- **响应式设计**: 支持桌面和移动设备

### 🎨 界面特色
- 深色主题，护眼舒适
- 流畅的动画效果
- 现代化的 UI 设计
- 直观的用户体验

### 📱 页面组成
1. **登录/注册页面**: 用户身份验证
2. **欢迎页面**: 建议问题卡片
3. **聊天界面**: 消息展示和输入
4. **侧边栏**: 对话历史管理
5. **模型管理界面**: AI模型配置管理（管理员专用）

## 技术实现

### 前端技术栈
- **HTML5**: 语义化标记
- **CSS3**: 现代样式和动画
- **Vanilla JavaScript**: 原生 JS 实现
- **Font Awesome**: 图标库

### API 对接
与后端 `api/v1/endpoints` 下的接口对接：

- `POST /auth/login` - 用户登录
- `POST /auth/register` - 用户注册
- `GET /conversations` - 获取对话列表
- `POST /conversations` - 创建新对话
- `GET /conversations/{id}/messages` - 获取对话消息
- `POST /messages/{conversation_id}/send` - 发送消息（流式响应）
- `GET /models` - 获取可用模型列表
- `GET /models/{model_id}` - 获取模型详细配置（管理员）
- `POST /models` - 创建新模型配置（管理员）
- `PUT /models/{model_id}` - 更新模型配置（管理员）

## 使用方法

### 1. 快速演示（推荐）
无需后端API，直接体验界面效果：
```bash
# 进入前端目录
cd frontend

# 使用任意 HTTP 服务器运行
python -m http.server 8080

# 在浏览器中访问演示页面
# http://localhost:8080/demo.html
```

### 2. 完整功能使用
需要配置后端API地址：
```bash
# 访问完整功能页面
# http://localhost:8080/index.html
```

### 3. 配置后端地址
修改 `script.js` 中的 API 地址：
```javascript
const CONFIG = {
    API_BASE_URL: 'http://your-backend-url/api/v1',
    // ...
};
```

### 3. 访问页面
打开浏览器访问：`http://localhost:8080`

## 文件结构

```
frontend/
├── index.html          # 主页面（完整功能）
├── demo.html           # 演示页面（无需后端）
├── styles.css          # 样式文件
├── script.js           # 脚本文件
└── README.md           # 说明文档
```

## 页面截图和功能说明

### 登录页面
- 用户名/邮箱 + 密码登录
- 支持切换到注册页面
- 表单验证和错误提示

### 欢迎页面
- 显示 Chat AI 品牌
- 提供建议问题卡片
- 点击卡片快速开始对话

### 聊天界面
- 左侧：对话历史列表
- 右侧：消息展示区域
- 底部：消息输入框
- 支持 Markdown 格式渲染

### 消息功能
- 实时流式响应
- 消息格式化显示
- 字符计数和限制
- 自动滚动到底部

### 对话管理
- 新建对话
- 重命名对话
- 删除对话
- 对话历史保存

### 模型管理（管理员功能）
- 查看所有可用模型
- 添加新的AI模型配置
- 编辑模型参数和状态
- 查看模型详细信息
- 启用/禁用模型

## 兼容性

### 浏览器支持
- Chrome 70+
- Firefox 65+
- Safari 12+
- Edge 79+

### 移动端适配
- 响应式布局
- 触摸友好
- 适配小屏幕

## 自定义配置

### 主题颜色
在 `styles.css` 中修改 CSS 变量：
```css
:root {
    --accent-color: #10a37f;    /* 主题色 */
    --bg-primary: #343541;      /* 主背景 */
    --text-primary: #ececf1;    /* 主文字 */
    /* ... */
}
```

### API 配置
在 `script.js` 中修改配置：
```javascript
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000/api/v1',
    TOKEN_KEY: 'chat_ai_token',
    USER_KEY: 'chat_ai_user'
};
```

## 注意事项

1. **CORS 设置**: 确保后端配置了正确的 CORS 策略
2. **HTTPS**: 生产环境建议使用 HTTPS
3. **Token 存储**: 敏感信息使用安全的存储方式
4. **错误处理**: 页面包含了网络错误的降级处理

## 开发计划

### 待实现功能
- [ ] 对话导出功能
- [ ] 主题切换（明暗模式）
- [ ] 消息搜索功能
- [ ] 文件上传支持
- [ ] 语音输入支持

### 优化项目
- [ ] 代码分割和懒加载
- [ ] PWA 支持
- [ ] 更多动画效果
- [ ] 性能优化

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目采用 MIT 许可证。 