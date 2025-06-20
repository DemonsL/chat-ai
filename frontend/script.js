// 全局配置
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000/api/v1',
    TOKEN_KEY: 'chat_ai_token',
    USER_KEY: 'chat_ai_user'
};

// 全局状态
let currentUser = null;
let currentConversation = null;
let conversations = [];
let accessToken = null;
let isLoading = false;
let availableModels = [];
let isAdmin = false;
let currentMode = 'chat'; // 当前对话模式
let selectedTools = ['search']; // 选中的工具
let uploadedFiles = []; // 上传的文件 - 改为存储文件对象和上传状态

// DOM 元素
const elements = {};

// 初始化应用
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    checkAuthStatus();
    setupEventListeners();
});

// 初始化DOM元素引用
function initializeElements() {
    elements.loginModal = document.getElementById('login-modal');
    elements.registerModal = document.getElementById('register-modal');
    elements.mainApp = document.getElementById('main-app');
    elements.loading = document.getElementById('loading');
    elements.conversationList = document.getElementById('conversation-list');
    elements.welcomeScreen = document.getElementById('welcome-screen');
    elements.chatInterface = document.getElementById('chat-interface');
    elements.messagesContainer = document.getElementById('messages-container');
    elements.messageInput = document.getElementById('message-input');
    elements.sendButton = document.getElementById('send-button');
    elements.userName = document.getElementById('user-name');
    elements.charCount = document.getElementById('char-count');
    elements.currentModeText = document.getElementById('current-mode-text');
    elements.modelDropdown = document.getElementById('model-dropdown');
    elements.modelDropdownMenu = document.getElementById('model-dropdown-menu');
    elements.selectedModelName = document.getElementById('selected-model-name');
    elements.fileInput = document.getElementById('file-input');
    elements.uploadedFiles = document.getElementById('uploaded-files');
    elements.fileUploadArea = document.getElementById('file-upload-area');
    elements.toolsMenu = document.getElementById('tools-menu');
    elements.attachMenu = document.getElementById('attach-menu');
}

// 检查用户认证状态
function checkAuthStatus() {
    const token = localStorage.getItem(CONFIG.TOKEN_KEY);
    const user = localStorage.getItem(CONFIG.USER_KEY);
    
    if (token && user) {
        accessToken = token;
        currentUser = JSON.parse(user);
        showMainApp();
    } else {
        showLoginModal();
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 登录表单
    document.getElementById('login-form').addEventListener('submit', handleLogin);
    
    // 注册表单
    document.getElementById('register-form').addEventListener('submit', handleRegister);
    
    // 消息输入
    elements.messageInput.addEventListener('input', handleInputChange);
    elements.messageInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 文件上传
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileUpload);
    }
    
    // 文件移除按钮事件委托
    document.addEventListener('click', (e) => {
        if (e.target.closest('.remove-file')) {
            const button = e.target.closest('.remove-file');
            const fileId = button.getAttribute('data-file-id');
            if (fileId) {
                removeFile(fileId);
            }
        }
    });
    
    // 点击其他地方关闭下拉菜单
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.model-dropdown')) {
            hideModelDropdown();
        }
        if (!e.target.closest('.tools-selector')) {
            hideToolsMenu();
        }
        if (!e.target.closest('.attach-menu') && !e.target.closest('.attach-btn')) {
            hideAttachMenu();
        }
    });
}

// 处理登录
async function handleLogin(e) {
    e.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showMessage('请填写用户名和密码', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('username', username);
        formData.append('password', password);
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/login`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const data = await response.json();
            accessToken = data.access_token;
            
            // 模拟用户信息 - 默认设置为管理员用于测试
            currentUser = { 
                username: username, 
                id: '1',
                is_admin: true  // 默认设置为管理员
            };
            
            // 保存到本地存储
            localStorage.setItem(CONFIG.TOKEN_KEY, accessToken);
            localStorage.setItem(CONFIG.USER_KEY, JSON.stringify(currentUser));
            
            hideLoading();
            closeModal('login-modal');
            showMainApp();
        } else {
            const error = await response.json();
            throw new Error(error.detail || '登录失败');
        }
    } catch (error) {
        hideLoading();
        showMessage(error.message, 'error');
    }
}

// 处理注册
async function handleRegister(e) {
    e.preventDefault();
    
    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const fullName = document.getElementById('reg-fullname').value;
    const password = document.getElementById('reg-password').value;
    
    if (!username || !email || !password) {
        showMessage('请填写必要信息', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/auth/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username,
                email,
                password,
                full_name: fullName || null
            })
        });
        
        if (response.ok) {
            hideLoading();
            closeModal('register-modal');
            showMessage('注册成功，请登录', 'success');
            showLoginModal();
        } else {
            const error = await response.json();
            throw new Error(error.detail || '注册失败');
        }
    } catch (error) {
        hideLoading();
        showMessage(error.message, 'error');
    }
}

// 显示主应用
async function showMainApp() {
    elements.mainApp.classList.remove('hidden');
    elements.userName.textContent = currentUser.username || currentUser.full_name;
    
    // 检查是否是管理员
    isAdmin = currentUser.is_admin || false;
    console.log('当前用户是管理员:', isAdmin); // 调试信息
    
    // 更新管理员菜单显示
    updateAdminMenu();
    
    await Promise.all([
        loadConversations(),
        loadAvailableModels()
    ]);
    
    // 更新模型选择器
    updateModelSelector();
    showWelcomeScreen();
}

// 更新管理员菜单显示
function updateAdminMenu() {
    console.log('更新管理员菜单，isAdmin:', isAdmin);
    const managementMenu = document.getElementById('management-menu');
    if (managementMenu) {
        if (isAdmin) {
            managementMenu.style.display = 'block';
            console.log('显示管理员菜单');
        } else {
            managementMenu.style.display = 'none';
            console.log('隐藏管理员菜单');
        }
    } else {
        console.log('找不到管理员菜单元素');
    }
}

// 加载对话列表
async function loadConversations() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/conversations`, {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });
        
        if (response.ok) {
            conversations = await response.json();
            renderConversationList();
        } else {
            console.log('获取对话列表失败，使用空列表');
            conversations = [];
            renderConversationList();
        }
    } catch (error) {
        console.log('网络错误，使用空对话列表');
        conversations = [];
        renderConversationList();
    }
}

// 渲染对话列表
function renderConversationList() {
    elements.conversationList.innerHTML = '';
    
    conversations.forEach(conversation => {
        const item = document.createElement('div');
        item.className = 'conversation-item';
        item.dataset.id = conversation.id;
        
        if (currentConversation && currentConversation.id === conversation.id) {
            item.classList.add('active');
        }
        
        item.innerHTML = `
            <span class="conversation-title">${conversation.title || '新对话'}</span>
            <div class="conversation-actions">
                <button class="conversation-action" onclick="renameConversation('${conversation.id}')" title="重命名">
                    <i class="fas fa-edit"></i>
                </button>
                <button class="conversation-action" onclick="deleteConversation('${conversation.id}')" title="删除">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        item.addEventListener('click', () => loadConversation(conversation.id));
        elements.conversationList.appendChild(item);
    });
}

// 创建新对话
async function createNewConversation() {
    const selectedModelId = getSelectedModelId();
    const newConversation = {
        id: Date.now().toString(),
        title: '新对话',
        created_at: new Date().toISOString(),
        mode: currentMode,
        model_id: selectedModelId
    };
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/conversations`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                title: '新对话',
                model_id: selectedModelId,
                mode: currentMode,
                file_ids: []
            })
        });
        
        if (response.ok) {
            const conversation = await response.json();
            conversations.unshift(conversation);
            currentConversation = conversation;
        } else {
            conversations.unshift(newConversation);
            currentConversation = newConversation;
        }
        
        renderConversationList();
        showChatInterface();
    } catch (error) {
        conversations.unshift(newConversation);
        currentConversation = newConversation;
        renderConversationList();
        showChatInterface();
    }
}

// 加载对话
async function loadConversation(conversationId) {
    try {
        showLoading();
        
        const conversation = conversations.find(conv => conv.id === conversationId);
        if (conversation) {
            currentConversation = conversation;
        }
        
        let messages = [];
        try {
            const msgResponse = await fetch(`${CONFIG.API_BASE_URL}/conversations/${conversationId}/messages`, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`
                }
            });
            
            if (msgResponse.ok) {
                messages = await msgResponse.json();
            }
        } catch (e) {
            console.log('获取消息失败，使用空消息列表');
        }
        
        hideLoading();
        renderConversationList();
        showChatInterface();
        renderMessages(messages);
        
    } catch (error) {
        hideLoading();
        showMessage(error.message, 'error');
    }
}

// 显示聊天界面
function showChatInterface() {
    elements.welcomeScreen.classList.add('hidden');
    elements.chatInterface.classList.remove('hidden');
    elements.messageInput.focus();
}

// 显示欢迎界面
function showWelcomeScreen() {
    elements.chatInterface.classList.add('hidden');
    elements.welcomeScreen.classList.remove('hidden');
    currentConversation = null;
    renderConversationList();
}

// 渲染消息
function renderMessages(messages) {
    elements.messagesContainer.innerHTML = '';
    
    messages.forEach(message => {
        addMessageToUI(message);
    });
    
    scrollToBottom();
}

// 添加消息到UI
function addMessageToUI(message) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = message.role === 'user' ? 
        '<i class="fas fa-user"></i>' : 
        '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = `<p>${formatMessageContent(message.content)}</p>`;
    
    messageEl.appendChild(avatar);
    messageEl.appendChild(content);
    
    elements.messagesContainer.appendChild(messageEl);
}

// 格式化消息内容
function formatMessageContent(content) {
    return content
        .replace(/```(\w+)?\n([\s\S]*?)\n```/g, '<pre><code>$2</code></pre>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

// 发送消息
async function sendMessage() {
    const content = elements.messageInput.value.trim();
    
    if (!content || isLoading) {
        return;
    }
    
    if (!currentConversation) {
        await createNewConversation();
    }
    
    if (!currentConversation) {
        showMessage('创建对话失败', 'error');
        return;
    }
    
    // 获取当前选中的模型
    const selectedModelId = getSelectedModelId();
    
    // 获取已成功上传和处理的文件ID
    const indexedFiles = uploadedFiles.filter(f => f.status === 'indexed' && f.fileId);
    const fileIds = indexedFiles.map(f => f.fileId);
    
    // 准备消息元数据
    const metadata = {
        mode: currentMode,
        model_id: selectedModelId,
        tools: currentMode === 'search' || currentMode === 'deepresearch' ? getSelectedTools() : [],
        file_ids: currentMode === 'rag' ? fileIds : [] // 直接传递文件ID
    };
    
    elements.messageInput.value = '';
    updateCharCount();
    
    addMessageToUI({
        role: 'user',
        content: content,
        metadata: metadata
    });
    
    showTypingIndicator();
    
    isLoading = true;
    elements.sendButton.disabled = true;
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/messages/${currentConversation.id}/send`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                content: content,
                metadata: metadata
            })
        });
        
        hideTypingIndicator();
        
        if (response.ok && response.body) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let assistantMessage = '';
            let messageEl = null;
            let sources = [];
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.error) {
                                throw new Error(data.message || '处理消息时发生错误');
                            }
                            
                            if (data.content) {
                                assistantMessage += data.content;
                                
                                if (!messageEl) {
                                    messageEl = createAssistantMessage();
                                }
                                
                                updateAssistantMessage(messageEl, assistantMessage);
                            }
                            
                            // 处理来源信息
                            if (data.sources) {
                                sources = data.sources;
                            }
                            
                            // 处理工具使用步骤
                            if (data.is_tool_use) {
                                showToolUseStep(messageEl, data);
                            }
                            
                            if (data.done) {
                                // 如果有来源信息，添加到消息中
                                if (sources.length > 0 && messageEl) {
                                    addSourcesToMessage(messageEl, sources);
                                }
                                break;
                            }
                        } catch (e) {
                            console.error('解析流数据失败:', e);
                        }
                    }
                }
            }
        } else {
            setTimeout(() => {
                const demoContent = getDemoResponse();
                addMessageToUI({
                    role: 'assistant',
                    content: demoContent
                });
                scrollToBottom();
            }, 1000);
        }
        
    } catch (error) {
        hideTypingIndicator();
        
        setTimeout(() => {
            addMessageToUI({
                role: 'assistant',
                content: '抱歉，我现在无法连接到服务器。这是一个模拟回复用于演示界面功能。请检查网络连接或稍后重试。'
            });
            scrollToBottom();
        }, 1000);
    } finally {
        isLoading = false;
        elements.sendButton.disabled = false;
        scrollToBottom();
    }
}

// 获取当前选中的模型ID
function getSelectedModelId() {
    const selectedOption = document.querySelector('.model-option.selected');
    if (selectedOption) {
        return selectedOption.getAttribute('data-model-id');
    }
    // 默认返回第一个可用模型
    return availableModels.length > 0 ? (availableModels[0].id || availableModels[0].model_id) : 'gpt-3.5-turbo';
}

// 显示工具使用步骤
function showToolUseStep(messageElement, data) {
    if (!messageElement) return;
    
    const toolStepEl = document.createElement('div');
    toolStepEl.className = 'tool-use-step';
    toolStepEl.innerHTML = `
        <div class="tool-use-header">
            <i class="fas fa-cog fa-spin"></i>
            <span>${data.tool_name || '使用工具'}</span>
        </div>
        <div class="tool-use-content">${data.content || ''}</div>
    `;
    
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.appendChild(toolStepEl);
        scrollToBottom();
    }
}

// 添加来源信息到消息
function addSourcesToMessage(messageElement, sources) {
    if (!sources || sources.length === 0 || !messageElement) return;
    
    const sourcesEl = document.createElement('div');
    sourcesEl.className = 'message-sources';
    sourcesEl.innerHTML = `
        <h4>参考来源:</h4>
        ${sources.map(source => `
            <div class="source-item">
                <a href="${source.url || '#'}" target="_blank" rel="noopener noreferrer">
                    ${source.title || source.filename || '未知来源'}
                </a>
                ${source.snippet ? `<p>${source.snippet}</p>` : ''}
            </div>
        `).join('')}
    `;
    
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.appendChild(sourcesEl);
    }
}

// 获取演示回复内容
function getDemoResponse() {
    const responses = {
        'chat': '这是一个模拟的AI回复。由于无法连接到后端API，我使用了一个示例响应。在实际部署时，这里将显示真正的AI生成内容。',
        'rag': '基于您上传的文档，我找到了相关信息。这是一个模拟的文档问答回复，实际使用时会根据文档内容提供准确答案。',
        'search': '我已经搜索了相关信息。这是一个模拟的搜索结果回复，实际使用时会提供最新的网络搜索结果。',
        'deepresearch': '我正在进行深度研究分析。这是一个模拟的研究回复，实际使用时会提供详细的分析和见解。'
    };
    
    return responses[currentMode] || responses['chat'];
}

// 创建助手消息元素
function createAssistantMessage() {
    const messageEl = document.createElement('div');
    messageEl.className = 'message assistant';
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const content = document.createElement('div');
    content.className = 'message-content';
    
    messageEl.appendChild(avatar);
    messageEl.appendChild(content);
    
    elements.messagesContainer.appendChild(messageEl);
    
    return messageEl;
}

// 更新助手消息内容
function updateAssistantMessage(messageEl, content) {
    const contentEl = messageEl.querySelector('.message-content');
    contentEl.innerHTML = `<p>${formatMessageContent(content)}</p>`;
    scrollToBottom();
}

// 显示打字指示器
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message assistant typing-message';
    indicator.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    elements.messagesContainer.appendChild(indicator);
    scrollToBottom();
}

// 隐藏打字指示器
function hideTypingIndicator() {
    const indicator = elements.messagesContainer.querySelector('.typing-message');
    if (indicator) {
        indicator.remove();
    }
}

// 发送建议消息
async function sendSuggestion(suggestion) {
    elements.messageInput.value = suggestion;
    await sendMessage();
}

// 处理输入变化
function handleInputChange() {
    updateCharCount();
    autoResizeTextarea();
    
    // 动态启用/禁用发送按钮
    const hasContent = elements.messageInput.value.trim().length > 0;
    elements.sendButton.disabled = !hasContent || isLoading;
}

// 更新字符计数
function updateCharCount() {
    const count = elements.messageInput.value.length;
    elements.charCount.textContent = `${count}/10000`;
}

// 自动调整文本框高度
function autoResizeTextarea() {
    elements.messageInput.style.height = 'auto';
    elements.messageInput.style.height = Math.min(elements.messageInput.scrollHeight, 120) + 'px';
}

// 滚动到底部
function scrollToBottom() {
    setTimeout(() => {
        elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
    }, 100);
}

// 清空对话
async function clearConversation() {
    if (!currentConversation) return;
    
    if (confirm('确定要清空当前对话吗？')) {
        elements.messagesContainer.innerHTML = '';
    }
}

// 删除对话
async function deleteConversation(conversationId, showConfirm = true) {
    if (showConfirm && !confirm('确定要删除这个对话吗？')) {
        return;
    }
    
    conversations = conversations.filter(conv => conv.id !== conversationId);
    
    if (currentConversation && currentConversation.id === conversationId) {
        showWelcomeScreen();
    }
    
    renderConversationList();
}

// 重命名对话
async function renameConversation(conversationId) {
    const conversation = conversations.find(conv => conv.id === conversationId);
    if (!conversation) return;
    
    const newTitle = prompt('请输入新的标题:', conversation.title);
    if (!newTitle || newTitle === conversation.title) return;
    
    conversation.title = newTitle;
    if (currentConversation && currentConversation.id === conversationId) {
        currentConversation.title = newTitle;
        elements.conversationTitle.textContent = newTitle;
    }
    renderConversationList();
}

// 分享对话
function shareConversation() {
    if (!currentConversation) return;
    
    const url = `${window.location.origin}?conversation=${currentConversation.id}`;
    navigator.clipboard.writeText(url).then(() => {
        showMessage('对话链接已复制到剪贴板', 'success');
    }).catch(() => {
        showMessage('复制失败，请手动复制链接', 'error');
    });
}

// 退出登录
function logout() {
    if (confirm('确定要退出登录吗？')) {
        localStorage.removeItem(CONFIG.TOKEN_KEY);
        localStorage.removeItem(CONFIG.USER_KEY);
        accessToken = null;
        currentUser = null;
        currentConversation = null;
        conversations = [];
        
        elements.mainApp.classList.add('hidden');
        showLoginModal();
    }
}

// 模态框操作
function showLoginModal() {
    elements.loginModal.classList.remove('hidden');
    document.getElementById('username').focus();
}

function showRegisterModal() {
    elements.registerModal.classList.remove('hidden');
    document.getElementById('reg-username').focus();
}

function switchToRegister() {
    closeModal('login-modal');
    showRegisterModal();
}

function switchToLogin() {
    closeModal('register-modal');
    showLoginModal();
}

// 模型管理功能

// 加载可用模型列表
async function loadAvailableModels() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/models`, {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });
        
        if (response.ok) {
            availableModels = await response.json();
        } else {
            console.log('获取模型列表失败，使用默认模型');
            availableModels = [
                { model_id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai', capabilities: ['chat'] },
                { model_id: 'gpt-4', name: 'GPT-4', provider: 'openai', capabilities: ['chat'] },
                { model_id: 'gpt-4-vision', name: 'GPT-4 Vision', provider: 'openai', capabilities: ['chat', 'vision'] },
                { model_id: 'gemini-pro', name: 'Gemini Pro', provider: 'google_genai', capabilities: ['chat'] },
                { model_id: 'gemini-pro-vision', name: 'Gemini Pro Vision', provider: 'google_genai', capabilities: ['chat', 'vision'] }
            ];
        }
    } catch (error) {
        console.log('网络错误，使用默认模型列表');
        availableModels = [
            { model_id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', provider: 'openai', capabilities: ['chat'] },
            { model_id: 'gpt-4', name: 'GPT-4', provider: 'openai', capabilities: ['chat'] },
            { model_id: 'gpt-4-vision', name: 'GPT-4 Vision', provider: 'openai', capabilities: ['chat', 'vision'] },
            { model_id: 'gemini-pro', name: 'Gemini Pro', provider: 'google_genai', capabilities: ['chat'] },
            { model_id: 'gemini-pro-vision', name: 'Gemini Pro Vision', provider: 'google_genai', capabilities: ['chat', 'vision'] }
        ];
    }
}

// 更新模型选择器
function updateModelSelector() {
    console.log('更新模型选择器，可用模型数量:', availableModels.length);
    
    if (!elements.modelDropdownMenu) {
        console.log('modelDropdownMenu元素不存在');
        return;
    }
    
    elements.modelDropdownMenu.innerHTML = '';
    
    if (availableModels.length === 0) {
        const option = document.createElement('div');
        option.className = 'model-option';
        option.innerHTML = '<div class="model-option-name">暂无可用模型</div>';
        elements.modelDropdownMenu.appendChild(option);
        return;
    }

    availableModels.forEach((model, index) => {
        const modelId = model.id || model.model_id;
        const modelName = model.name || model.display_name || modelId;
        const capabilities = model.capabilities || [];
        
        const option = document.createElement('div');
        option.className = 'model-option';
        option.setAttribute('data-model-id', modelId);
        option.onclick = () => selectModel(modelId, modelName);
        
        option.innerHTML = `
            <div class="model-option-header">
                <div class="model-option-name">${modelName}</div>
                ${model.is_active !== false ? '<div class="model-option-badge">可用</div>' : ''}
            </div>
            <div class="model-option-description">
                ${model.provider || '未知提供商'} • ${capabilities.join(', ') || '通用模型'}
            </div>
        `;
        
        // 设置第一个模型为默认选中
        if (index === 0) {
            option.classList.add('selected');
            elements.selectedModelName.textContent = modelName;
        }
        
        elements.modelDropdownMenu.appendChild(option);
    });
    
    console.log('模型选择器已更新，选项数量:', availableModels.length);
}

// 获取模型详细信息（管理员功能）
async function getModelDetails(modelId) {
    if (!isAdmin) {
        showMessage('需要管理员权限', 'error');
        return null;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/models/${modelId}`, {
            headers: {
                'Authorization': `Bearer ${accessToken}`
            }
        });
        
        if (response.ok) {
            return await response.json();
        } else {
            throw new Error('获取模型详情失败');
        }
    } catch (error) {
        showMessage(error.message, 'error');
        return null;
    }
}

// 创建新模型配置（管理员功能）
async function createModel(modelConfig) {
    if (!isAdmin) {
        showMessage('需要管理员权限', 'error');
        return false;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/models`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(modelConfig)
        });
        
        if (response.ok) {
            const newModel = await response.json();
            // 转换字段格式以保持一致性（ModelConfigResponse -> ModelInfo）
            availableModels.push({
                id: newModel.model_id,
                name: newModel.display_name,
                provider: newModel.provider,
                capabilities: newModel.capabilities,
                max_tokens: newModel.max_tokens,
                is_active: newModel.is_active
            });
            updateModelSelector();
            showMessage('模型创建成功', 'success');
            return true;
        } else {
            const error = await response.json();
            throw new Error(error.detail || '创建模型失败');
        }
    } catch (error) {
        showMessage(error.message, 'error');
        return false;
    }
}

// 更新模型配置（管理员功能）
async function updateModel(modelId, modelUpdate) {
    if (!isAdmin) {
        showMessage('需要管理员权限', 'error');
        return false;
    }
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/models/${modelId}`, {
            method: 'PUT',
            headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(modelUpdate)
        });
        
        if (response.ok) {
            const updatedModel = await response.json();
            // 注意：availableModels 使用 ModelInfo 格式 (id, name)，但更新返回 ModelConfigResponse 格式 (model_id, display_name)
            const index = availableModels.findIndex(model => (model.id || model.model_id) === modelId);
            if (index !== -1) {
                // 转换字段格式以保持一致性
                availableModels[index] = {
                    id: updatedModel.model_id,
                    name: updatedModel.display_name,
                    provider: updatedModel.provider,
                    capabilities: updatedModel.capabilities,
                    max_tokens: updatedModel.max_tokens,
                    is_active: updatedModel.is_active
                };
                updateModelSelector();
            }
            showMessage('模型更新成功', 'success');
            return true;
        } else {
            const error = await response.json();
            throw new Error(error.detail || '更新模型失败');
        }
    } catch (error) {
        showMessage(error.message, 'error');
        return false;
    }
}

// 显示模型管理界面
function showModelManagement() {
    console.log('showModelManagement 函数被调用');
    console.log('当前用户是管理员:', isAdmin);
    
    if (!isAdmin) {
        console.log('用户不是管理员，显示错误消息');
        showMessage('需要管理员权限访问模型管理', 'error');
        return;
    }
    
    console.log('用户是管理员，继续显示模型管理界面');
    
    // 隐藏其他界面
    elements.welcomeScreen.classList.add('hidden');
    elements.chatInterface.classList.add('hidden');
    console.log('已隐藏其他界面');
    
    // 创建模型管理界面
    let modelManagementEl = document.getElementById('model-management');
    console.log('查找现有模型管理元素:', modelManagementEl);
    
    if (!modelManagementEl) {
        console.log('创建新的模型管理界面');
        modelManagementEl = document.createElement('div');
        modelManagementEl.id = 'model-management';
        modelManagementEl.className = 'model-management';
        
        modelManagementEl.innerHTML = `
            <div class="management-header">
                <h2>模型管理</h2>
                <div class="management-actions">
                    <button class="btn btn-primary" id="add-model-btn">
                        <i class="fas fa-plus"></i>
                        添加模型
                    </button>
                    <button class="btn btn-secondary" id="refresh-models-btn">
                        <i class="fas fa-refresh"></i>
                        刷新
                    </button>
                </div>
            </div>
            
            <div class="models-grid" id="models-grid">
                <!-- 模型卡片将在这里动态生成 -->
            </div>
        `;
        
        const chatArea = document.querySelector('.chat-area');
        console.log('找到聊天区域:', chatArea);
        if (chatArea) {
            chatArea.appendChild(modelManagementEl);
            console.log('模型管理界面已添加到聊天区域');
            
            // 使用addEventListener绑定事件，更可靠
            const addModelBtn = document.getElementById('add-model-btn');
            const refreshModelsBtn = document.getElementById('refresh-models-btn');
            
            if (addModelBtn) {
                console.log('绑定添加模型按钮事件');
                addModelBtn.addEventListener('click', function() {
                    console.log('添加模型按钮被点击');
                    showCreateModelModal();
                });
            }
            
            if (refreshModelsBtn) {
                console.log('绑定刷新模型按钮事件');
                refreshModelsBtn.addEventListener('click', function() {
                    console.log('刷新模型按钮被点击');
                    refreshModels();
                });
            }
        } else {
            console.error('找不到聊天区域元素');
            return;
        }
    }
    
    console.log('显示模型管理界面');
    modelManagementEl.classList.remove('hidden');
    
    console.log('渲染模型网格，可用模型数量:', availableModels.length);
    renderModelsGrid();
    
    console.log('模型管理界面设置完成');
}

// 渲染模型网格
function renderModelsGrid() {
    const gridEl = document.getElementById('models-grid');
    if (!gridEl) return;
    
    gridEl.innerHTML = '';
    
    availableModels.forEach(model => {
        const modelCard = document.createElement('div');
        modelCard.className = 'model-card';
        
        const capabilities = model.capabilities || [];
        const capabilitiesText = capabilities.join(', ');
        
        // ModelInfo 格式的字段：id, name, provider, capabilities, max_tokens
        const modelId = model.id || model.model_id;
        const modelName = model.name || model.display_name;
        const providerName = model.provider;
        
        modelCard.innerHTML = `
            <div class="model-card-header">
                <h3>${modelName || modelId}</h3>
                <div class="model-status ${model.is_active !== false ? 'active' : 'inactive'}">
                    ${model.is_active !== false ? '启用' : '禁用'}
                </div>
            </div>
            
            <div class="model-card-body">
                <p><strong>模型ID:</strong> ${modelId}</p>
                <p><strong>提供商:</strong> ${providerName || '未知'}</p>
                <p><strong>能力:</strong> ${capabilitiesText}</p>
                <p><strong>最大令牌数:</strong> ${model.max_tokens || '未设置'}</p>
            </div>
            
            <div class="model-card-actions">
                <button class="btn btn-secondary" onclick="editModel('${modelId}')">
                    <i class="fas fa-edit"></i>
                    编辑
                </button>
                <button class="btn btn-secondary" onclick="viewModelDetails('${modelId}')">
                    <i class="fas fa-eye"></i>
                    详情
                </button>
            </div>
        `;
        
        gridEl.appendChild(modelCard);
    });
}

// 显示创建模型模态框
function showCreateModelModal() {
    console.log('showCreateModelModal 函数被调用');
    console.log('当前用户是管理员:', isAdmin);
    
    // 添加权限检查（虽然按钮只有管理员才能看到，但为了安全起见）
    if (!isAdmin) {
        showMessage('需要管理员权限', 'error');
        return;
    }
    
    console.log('开始创建模态框');
    
    // 临时隐藏模型管理界面，避免层级冲突
    const modelManagement = document.getElementById('model-management');
    if (modelManagement) {
        modelManagement.style.display = 'none';
        console.log('临时隐藏模型管理界面');
    }
    
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>添加新模型</h2>
                <button class="close-btn" onclick="closeCreateModelModal(this)">&times;</button>
            </div>
            
            <form id="create-model-form" class="model-form">
                <div class="form-row">
                    <div class="form-group">
                        <label>模型ID *</label>
                        <input type="text" name="model_id" required placeholder="例: gpt-4-turbo">
                    </div>
                    <div class="form-group">
                        <label>模型名称 *</label>
                        <input type="text" name="name" required placeholder="例: GPT-4 Turbo">
                    </div>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>提供商 *</label>
                        <select name="provider_name" required>
                            <option value="">选择提供商</option>
                            <option value="openai">OpenAI</option>
                            <option value="anthropic">Anthropic</option>
                            <option value="google_genai">Google</option>
                            <option value="deepseek">DeepSeek</option>
                            <option value="other">其他</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>最大令牌数 *</label>
                        <input type="number" name="max_tokens" required min="1" max="100000" value="4000" placeholder="4000">
                    </div>
                </div>
                
                <div class="form-group">
                    <label>能力 *</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" name="capabilities" value="chat" checked> 聊天</label>
                        <label><input type="checkbox" name="capabilities" value="rag"> RAG</label>
                        <label><input type="checkbox" name="capabilities" value="agent"> 智能体</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>描述</label>
                    <textarea name="description" rows="3" placeholder="模型描述..."></textarea>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label>API密钥</label>
                        <input type="password" name="api_key" placeholder="API密钥">
                    </div>
                    <div class="form-group">
                        <label>API基础URL</label>
                        <input type="url" name="api_base" placeholder="例: https://api.openai.com/v1">
                    </div>
                </div>
                
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary" onclick="closeCreateModelModal(this)">
                        取消
                    </button>
                    <button type="submit" class="btn btn-primary">
                        创建模型
                    </button>
                </div>
            </form>
        </div>
    `;
    
    console.log('模态框HTML已创建，准备添加到页面');
    document.body.appendChild(modal);
    console.log('模态框已添加到页面，应该现在可见了');
    
    // 处理表单提交
    const form = document.getElementById('create-model-form');
    if (form) {
        console.log('找到表单，添加事件监听器');
        form.addEventListener('submit', async function(e) {
            console.log('表单提交事件触发');
            e.preventDefault();
            
            const formData = new FormData(this);
            const capabilities = Array.from(formData.getAll('capabilities'));
            
            const modelConfig = {
                model_id: formData.get('model_id'),
                display_name: formData.get('name'),
                provider: formData.get('provider_name'),
                capabilities: capabilities,
                max_tokens: parseInt(formData.get('max_tokens')) || 4000,
                is_active: true,
                config: {
                    api_key: formData.get('api_key'),
                    api_base: formData.get('api_base'),
                    // description: formData.get('description')
                }
            };
            
            console.log('准备创建模型:', modelConfig);
            const success = await createModel(modelConfig);
            if (success) {
                modal.remove();
                console.log('模型创建成功，模态框已关闭');
                
                // 恢复模型管理界面的显示
                const modelManagement = document.getElementById('model-management');
                if (modelManagement) {
                    modelManagement.style.display = 'flex';
                    console.log('恢复模型管理界面显示');
                }
                
                renderModelsGrid();
            }
        });
    } else {
        console.error('找不到表单元素');
    }
}

// 编辑模型
async function editModel(modelId) {
    const modelDetails = await getModelDetails(modelId);
    if (!modelDetails) return;
    
    // 创建编辑模态框（类似创建模态框，但预填充数据）
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content">
            <div class="modal-header">
                <h2>编辑模型: ${modelDetails.display_name || modelDetails.model_id}</h2>
                <button class="close-btn" onclick="this.closest('.modal').remove()">&times;</button>
            </div>
            
            <form id="edit-model-form" class="model-form">
                <div class="form-row">
                    <div class="form-group">
                        <label>模型名称</label>
                        <input type="text" name="display_name" value="${modelDetails.display_name || ''}" placeholder="例: GPT-4 Turbo">
                    </div>
                    <div class="form-group">
                        <label>状态</label>
                        <select name="is_active">
                            <option value="true" ${modelDetails.is_active ? 'selected' : ''}>启用</option>
                            <option value="false" ${!modelDetails.is_active ? 'selected' : ''}>禁用</option>
                        </select>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>最大令牌数</label>
                    <input type="number" name="max_tokens" value="${modelDetails.max_tokens || 4000}" min="1" max="100000">
                </div>
                
                <div class="form-group">
                    <label>API密钥环境变量</label>
                    <input type="text" name="api_key_env_name" value="${modelDetails.api_key_env_name || ''}" placeholder="例: OPENAI_API_KEY">
                </div>
                
                <div class="form-actions">
                    <button type="button" class="btn btn-secondary" onclick="this.closest('.modal').remove()">
                        取消
                    </button>
                    <button type="submit" class="btn btn-primary">
                        更新模型
                    </button>
                </div>
            </form>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // 处理表单提交
    document.getElementById('edit-model-form').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const modelUpdate = {
            display_name: formData.get('display_name'),
            max_tokens: parseInt(formData.get('max_tokens')) || 4000,
            api_key_env_name: formData.get('api_key_env_name'),
            is_active: formData.get('is_active') === 'true'
        };
        
        const success = await updateModel(modelId, modelUpdate);
        if (success) {
            modal.remove();
            renderModelsGrid();
        }
    });
}

// 查看模型详情
async function viewModelDetails(modelId) {
    const modelDetails = await getModelDetails(modelId);
    if (!modelDetails) return;
    
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.innerHTML = `
        <div class="modal-content model-details-modal">
            <div class="modal-header">
                <h2>模型详情</h2>
                <button class="close-btn" onclick="this.closest('.modal').remove()">&times;</button>
            </div>
            
            <div class="model-details">
                <div class="detail-item">
                    <strong>模型ID:</strong> ${modelDetails.model_id}
                </div>
                <div class="detail-item">
                    <strong>名称:</strong> ${modelDetails.display_name || '未设置'}
                </div>
                <div class="detail-item">
                    <strong>提供商:</strong> ${modelDetails.provider || '未知'}
                </div>
                <div class="detail-item">
                    <strong>最大令牌数:</strong> ${modelDetails.max_tokens || '未设置'}
                </div>
                <div class="detail-item">
                    <strong>能力:</strong> ${(modelDetails.capabilities || []).join(', ')}
                </div>
                <div class="detail-item">
                    <strong>状态:</strong> 
                    <span class="model-status ${modelDetails.is_active ? 'active' : 'inactive'}">
                        ${modelDetails.is_active ? '启用' : '禁用'}
                    </span>
                </div>
                <div class="detail-item">
                    <strong>API密钥环境变量:</strong> ${modelDetails.api_key_env_name || '未设置'}
                </div>
                <div class="detail-item">
                    <strong>配置:</strong> ${modelDetails.config ? JSON.stringify(modelDetails.config, null, 2) : '无配置'}
                </div>
                <div class="detail-item">
                    <strong>创建时间:</strong> ${modelDetails.created_at ? new Date(modelDetails.created_at).toLocaleString() : '未知'}
                </div>
                <div class="detail-item">
                    <strong>更新时间:</strong> ${modelDetails.updated_at ? new Date(modelDetails.updated_at).toLocaleString() : '未知'}
                </div>
            </div>
            
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="this.closest('.modal').remove()">
                    关闭
                </button>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
}

// 刷新模型列表
async function refreshModels() {
    showLoading();
    await loadAvailableModels();
    updateModelSelector();
    renderModelsGrid();
    hideLoading();
    showMessage('模型列表已刷新', 'success');
}

function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

function showLoading() {
    elements.loading.classList.remove('hidden');
}

function hideLoading() {
    elements.loading.classList.add('hidden');
}

function showMessage(message, type = 'info') {
    const messageEl = document.createElement('div');
    messageEl.className = `message-toast ${type}`;
    messageEl.textContent = message;
    
    Object.assign(messageEl.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '12px 16px',
        borderRadius: '4px',
        color: 'white',
        fontWeight: '500',
        zIndex: '1001',
        maxWidth: '300px',
        wordWrap: 'break-word'
    });
    
    switch (type) {
        case 'success':
            messageEl.style.backgroundColor = '#52c41a';
            break;
        case 'error':
            messageEl.style.backgroundColor = '#ff4d4f';
            break;
        case 'warning':
            messageEl.style.backgroundColor = '#faad14';
            break;
        default:
            messageEl.style.backgroundColor = '#10a37f';
    }
    
    document.body.appendChild(messageEl);
    
    setTimeout(() => {
        if (messageEl.parentNode) {
            messageEl.parentNode.removeChild(messageEl);
        }
    }, 3000);
}

// 模式选择功能
function selectMode(mode) {
    currentMode = mode;
    
    // 更新按钮状态
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
    
    // 更新模式文本
    const modeTexts = {
        'chat': '聊天模式',
        'rag': '文档问答',
        'search': '联网搜索',
        'deepresearch': '深度研究'
    };
    elements.currentModeText.textContent = modeTexts[mode] || '聊天模式';
    
    // 显示/隐藏文件上传区域
    if (mode === 'rag') {
        elements.fileUploadArea.style.display = 'flex';
    } else {
        elements.fileUploadArea.style.display = 'none';
    }
    
    // 更新工具选择器可见性
    const toolsSelector = document.getElementById('tools-selector');
    if (mode === 'search' || mode === 'deepresearch') {
        toolsSelector.style.display = 'flex';
    } else {
        toolsSelector.style.display = 'none';
    }
    
    console.log('切换到模式:', mode);
}

// 工具菜单切换
function toggleToolsMenu() {
    const menu = elements.toolsMenu;
    if (menu.classList.contains('show')) {
        hideToolsMenu();
    } else {
        showToolsMenu();
    }
}

function showToolsMenu() {
    elements.toolsMenu.classList.add('show');
}

function hideToolsMenu() {
    elements.toolsMenu.classList.remove('show');
}

// 附件菜单切换
function toggleAttachMenu() {
    const menu = elements.attachMenu;
    if (menu.classList.contains('show')) {
        hideAttachMenu();
    } else {
        showAttachMenu();
    }
}

function showAttachMenu() {
    elements.attachMenu.classList.add('show');
}

function hideAttachMenu() {
    elements.attachMenu.classList.remove('show');
}

// 模型下拉菜单
function toggleModelDropdown() {
    const menu = elements.modelDropdownMenu;
    if (menu.classList.contains('show')) {
        hideModelDropdown();
    } else {
        showModelDropdown();
    }
}

function showModelDropdown() {
    elements.modelDropdownMenu.classList.add('show');
}

function hideModelDropdown() {
    elements.modelDropdownMenu.classList.remove('show');
}

// 选择模型
function selectModel(modelId, modelName) {
    elements.selectedModelName.textContent = modelName;
    
    // 更新选中状态
    document.querySelectorAll('.model-option').forEach(option => {
        option.classList.remove('selected');
    });
    document.querySelector(`[data-model-id="${modelId}"]`).classList.add('selected');
    
    hideModelDropdown();
    console.log('选择模型:', modelId, modelName);
}

// 文件上传处理
function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        // 检查文件类型
        const allowedTypes = ['.txt', '.pdf', '.doc', '.docx', '.md'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!allowedTypes.includes(fileExtension)) {
            showMessage(`不支持的文件类型: ${file.name}`, 'error');
            return;
        }
        
        // 检查文件大小 (10MB限制)
        if (file.size > 10 * 1024 * 1024) {
            showMessage(`文件过大: ${file.name}`, 'error');
            return;
        }
        
        // 创建文件对象并立即开始上传
        const fileObj = {
            id: Date.now() + Math.random(),
            file: file,
            name: file.name,
            size: file.size,
            type: file.type,
            status: 'uploading', // uploading, processing, indexed, error
            progress: 0,
            fileId: null, // 服务器返回的文件ID
            error: null
        };
        
        // 添加到上传文件列表
        uploadedFiles.push(fileObj);
        renderUploadedFiles();
        
        // 立即开始上传
        uploadFileImmediately(fileObj);
    });
    
    event.target.value = ''; // 清空input
}

// 立即上传单个文件
async function uploadFileImmediately(fileObj) {
    try {
        // 更新状态为上传中
        fileObj.status = 'uploading';
        fileObj.progress = 0;
        renderUploadedFiles();
        
        const formData = new FormData();
        formData.append('file', fileObj.file);
        formData.append('description', `用户上传的文件: ${fileObj.name}`);
        
        // 如果有当前对话，关联到对话
        if (currentConversation && currentConversation.id) {
            formData.append('conversation_id', currentConversation.id);
        }
        
        // 异步处理文件，不阻塞上传响应
        formData.append('sync_process', 'false');
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/files/upload`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${accessToken}`
            },
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            fileObj.fileId = result.id;
            fileObj.progress = 100;
            
            // 根据文件状态更新UI状态
            if (result.status === 'indexed') {
                fileObj.status = 'indexed';
            } else {
                fileObj.status = 'processing';
                // 开始轮询检查处理状态
                pollFileStatus(fileObj);
            }
            
            renderUploadedFiles();
            showMessage(`文件上传成功: ${fileObj.name}`, 'success');
        } else {
            const error = await response.json();
            fileObj.status = 'error';
            fileObj.error = error.detail || '上传失败';
            renderUploadedFiles();
            showMessage(`文件上传失败: ${fileObj.name} - ${fileObj.error}`, 'error');
        }
    } catch (error) {
        fileObj.status = 'error';
        fileObj.error = error.message || '网络错误';
        renderUploadedFiles();
        showMessage(`上传文件时出错: ${fileObj.name} - ${fileObj.error}`, 'error');
    }
}

// 轮询检查文件处理状态
async function pollFileStatus(fileObj) {
    if (!fileObj.fileId) return;
    
    const maxAttempts = 30; // 最多轮询30次（约30秒）
    let attempts = 0;
    
    const poll = async () => {
        if (attempts >= maxAttempts) {
            fileObj.status = 'error';
            fileObj.error = '处理超时';
            renderUploadedFiles();
            return;
        }
        
        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/files/${fileObj.fileId}/status`, {
                headers: {
                    'Authorization': `Bearer ${accessToken}`
                }
            });
            
            if (response.ok) {
                const result = await response.json();
                
                if (result.status === 'indexed') {
                    fileObj.status = 'indexed';
                    renderUploadedFiles();
                    showMessage(`文件处理完成: ${fileObj.name}`, 'success');
                } else if (result.status === 'error') {
                    fileObj.status = 'error';
                    fileObj.error = result.error_message || '处理失败';
                    renderUploadedFiles();
                    showMessage(`文件处理失败: ${fileObj.name}`, 'error');
                } else {
                    // 继续轮询
                    attempts++;
                    setTimeout(poll, 1000); // 1秒后再次检查
                }
            } else {
                attempts++;
                setTimeout(poll, 1000);
            }
        } catch (error) {
            attempts++;
            setTimeout(poll, 1000);
        }
    };
    
    // 开始轮询
    setTimeout(poll, 1000);
}

// 渲染已上传文件
function renderUploadedFiles() {
    if (!elements.uploadedFiles) return;
    
    elements.uploadedFiles.innerHTML = '';
    
    uploadedFiles.forEach(fileObj => {
        const fileEl = document.createElement('div');
        fileEl.className = 'uploaded-file';
        
        // 根据状态选择图标
        let statusIcon = '';
        let statusClass = '';
        let statusText = '';
        
        switch (fileObj.status) {
            case 'uploading':
                statusIcon = '<i class="fas fa-upload fa-spin"></i>';
                statusClass = 'status-uploading';
                statusText = `上传中 ${fileObj.progress}%`;
                break;
            case 'processing':
                statusIcon = '<i class="fas fa-cog fa-spin"></i>';
                statusClass = 'status-processing';
                statusText = '处理中...';
                break;
            case 'indexed':
                statusIcon = '<i class="fas fa-check-circle"></i>';
                statusClass = 'status-indexed';
                statusText = '已完成';
                break;
            case 'error':
                statusIcon = '<i class="fas fa-exclamation-circle"></i>';
                statusClass = 'status-error';
                statusText = fileObj.error || '处理失败';
                break;
            default:
                statusIcon = '<i class="fas fa-file"></i>';
                statusClass = '';
                statusText = '未知状态';
        }
        
        fileEl.innerHTML = `
            <div class="file-info">
                <div class="file-icon">
                    <i class="fas fa-file"></i>
                </div>
                <div class="file-details">
                    <span class="file-name">${fileObj.name}</span>
                    <span class="file-status ${statusClass}" title="${statusText}">
                        ${statusIcon}
                        <span class="status-text">${statusText}</span>
                    </span>
                </div>
            </div>
            <button class="remove-file" data-file-id="${fileObj.id}" title="移除文件">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        elements.uploadedFiles.appendChild(fileEl);
    });
}

// 移除文件
function removeFile(fileId) {
    const index = uploadedFiles.findIndex(f => f.id === fileId);
    if (index !== -1) {
        uploadedFiles.splice(index, 1);
        renderUploadedFiles();
        showMessage('文件已移除', 'success');
    }
}

// 上传文件按钮点击
function uploadFile() {
    elements.fileInput.click();
    hideAttachMenu();
}

// 创建图片功能（占位符）
function createImage() {
    showMessage('图片创建功能即将推出', 'info');
    hideAttachMenu();
}

// 获取选中的工具
function getSelectedTools() {
    const checkboxes = document.querySelectorAll('input[name="tool"]:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

// 测试函数 - 用于调试
function testCreateModelModal() {
    console.log('测试函数被调用');
    console.log('isAdmin:', isAdmin);
    console.log('availableModels:', availableModels);
    showCreateModelModal();
}

// 添加全局调试函数到window对象，方便在控制台调用
window.testCreateModelModal = testCreateModelModal;
window.showCreateModelModal = showCreateModelModal;
window.showModelManagement = showModelManagement;

// 关闭创建模型模态框的函数
function closeCreateModelModal(element) {
    const modal = element.closest('.modal');
    if (modal) {
        modal.remove();
        console.log('模态框已关闭');
        
        // 恢复模型管理界面的显示
        const modelManagement = document.getElementById('model-management');
        if (modelManagement) {
            modelManagement.style.display = 'flex';
            console.log('恢复模型管理界面显示');
        }
    }
}

// 将关闭函数也添加到全局
window.closeCreateModelModal = closeCreateModelModal;
