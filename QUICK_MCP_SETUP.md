# 🚀 MCP Server クイック設定ガイド

## 1分で始める MCP Server 設定

### ⚡ 基本セットアップ

```bash
# 1. EPUB-LLM を起動
python run_server.py

# 2. 別ターミナルで MCP Server を起動
python start_mcp.py

# 3. 動作確認
python examples/test_mcp_client.py
```

### 📱 Claude Desktop 設定

#### 1. 設定ファイルを開く
```bash
# macOS
open ~/Library/Application\ Support/Claude/

# Windows
explorer %APPDATA%\Claude\

# Linux  
nautilus ~/.config/Claude/
```

#### 2. claude_desktop_config.json を編集
```json
{
  "mcpServers": {
    "epub-llm": {
      "command": "python",
      "args": ["絶対パス/epub-llm/start_mcp.py"],
      "env": {
        "PYTHONPATH": "絶対パス/epub-llm"
      }
    }
  }
}
```

**重要**: `絶対パス/epub-llm` を実際のパスに置き換えてください

#### 3. Claude Desktop を再起動

### 🧪 動作確認

Claude Desktop で以下を試してください：

```
EPUBファイル一覧を取得してください
```

```
「sample.epub」から「Python」について検索してください
```

### 🔧 トラブルシューティング

#### MCP Server が起動しない
```bash
# 依存関係確認
pip install fastmcp uvicorn

# ポート確認
lsof -i :8001
```

#### Claude で認識されない
1. パスが正しいか確認
2. Claude Desktop を完全に再起動
3. 設定ファイルのJSONが正しいか確認

### 📚 詳細ガイド

より詳しい設定方法は [MCP Server 設定ガイド](docs/mcp-server-setup.md) を参照してください。