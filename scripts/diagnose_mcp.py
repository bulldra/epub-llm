#!/usr/bin/env python3
"""
MCP Server 接続問題診断スクリプト

このスクリプトは MCP Server の接続問題を診断し、
解決策を提示します。
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Python バージョン確認"""
    print("🐍 Python バージョン確認")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8 以上が必要です")
        return False
    else:
        print("   ✅ Python バージョン OK")
        return True


def check_dependencies():
    """依存関係確認"""
    print("\n📦 依存関係確認")
    required_packages = [
        "fastmcp",
        "uvicorn",
        "fastapi",
        "mlx_lm",
        "mlx",
        "faiss",
        "requests",
    ]

    all_ok = True
    for package in required_packages:
        try:
            if package == "mlx_lm":
                import mlx_lm  # noqa: F401
            elif package == "mlx":
                import mlx.core  # noqa: F401
            elif package == "faiss":
                import faiss  # noqa: F401
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - インストールが必要")
            all_ok = False

    return all_ok


def check_project_structure():
    """プロジェクト構造確認"""
    print("\n📁 プロジェクト構造確認")

    required_files = [
        "src/app.py",
        "src/mcp_server.py",
        "start_mcp.py",
    ]

    project_root = Path(__file__).resolve().parents[1]
    all_ok = True

    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - ファイルが見つかりません")
            all_ok = False

    return all_ok


def check_port_availability():
    """ポート使用状況確認"""
    print("\n🌐 ポート使用状況確認")

    try:
        import socket

        def check_port(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(("localhost", port))
                return result != 0  # 0でない場合は使用可能

        ports = [8000, 8001]
        for port in ports:
            if check_port(port):
                print(f"   ✅ ポート {port} は使用可能")
            else:
                print(f"   ⚠️ ポート {port} は使用中")

    except Exception as e:
        print(f"   ❌ ポート確認エラー: {e}")
        return False

    return True


def check_claude_config():
    """Claude Desktop 設定確認"""
    print("\n⚙️ Claude Desktop 設定確認")

    config_paths = [
        Path.home() / "Library/Application Support/Claude/claude_desktop_config.json",
        Path.home() / ".config/Claude/claude_desktop_config.json",
        Path(os.environ.get("APPDATA", "")) / "Claude/claude_desktop_config.json",
    ]

    config_found = False
    for config_path in config_paths:
        if config_path.exists():
            print(f"   ✅ 設定ファイル発見: {config_path}")
            config_found = True

            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)

                if "mcpServers" in config:
                    if "epub-llm" in config["mcpServers"]:
                        print("   ✅ epub-llm サーバー設定あり")

                        server_config = config["mcpServers"]["epub-llm"]
                        command = server_config.get("command", "")
                        args = server_config.get("args", [])

                        print(f"   コマンド: {command}")
                        print(f"   引数: {args}")

                        # スクリプトファイルの存在確認
                        if args:
                            script_path = Path(args[0])
                            if script_path.exists():
                                print(f"   ✅ スクリプトファイル存在: {script_path}")
                            else:
                                print(f"   ❌ スクリプトファイル不存在: {script_path}")
                    else:
                        print("   ❌ epub-llm サーバー設定なし")
                else:
                    print("   ❌ mcpServers 設定なし")

            except json.JSONDecodeError:
                print("   ❌ 設定ファイルのJSON形式が無効")
            except Exception as e:
                print(f"   ❌ 設定ファイル読み込みエラー: {e}")

            break

    if not config_found:
        print("   ❌ Claude Desktop 設定ファイルが見つかりません")
        print("   以下のパスを確認してください:")
        for path in config_paths:
            print(f"     - {path}")

    return config_found


def test_mcp_server():
    """MCP Server テスト起動"""
    print("\n🧪 MCP Server テスト起動")

    project_root = Path(__file__).resolve().parents[1]

    try:
        # 環境変数設定
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        env["DEV_MODE"] = "true"
        env["PYTHONUNBUFFERED"] = "1"

        # テスト起動（5秒でタイムアウト）
        cmd = [sys.executable, "start_mcp.py"]

        print(f"   コマンド実行: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=5)
            print("   ✅ MCP Server は正常に起動開始")

            if stdout:
                print(f"   標準出力: {stdout[:200]}...")
            if stderr:
                print(f"   エラー出力: {stderr[:200]}...")

        except subprocess.TimeoutExpired:
            process.terminate()
            print("   ✅ MCP Server は5秒間正常動作")

    except Exception as e:
        print(f"   ❌ MCP Server テスト起動エラー: {e}")
        return False

    return True


def provide_solutions():
    """解決策提示"""
    print("\n🔧 推奨解決策")

    print("1. 依存関係インストール:")
    print("   pip install fastmcp uvicorn fastapi requests")
    print("   pip install mlx-lm mlx faiss-cpu")

    print("\n2. Claude Desktop 設定:")
    print("   以下の内容で設定ファイルを作成:")
    print("   ~/Library/Application Support/Claude/claude_desktop_config.json")

    project_root = Path(__file__).resolve().parents[1].absolute()
    config_example = {
        "mcpServers": {
            "epub-llm": {
                "command": "python3",
                "args": [str(project_root / "start_mcp.py")],
                "env": {
                    "PYTHONPATH": str(project_root),
                    "DEV_MODE": "true",
                    "PYTHONUNBUFFERED": "1",
                },
            }
        }
    }

    print(json.dumps(config_example, indent=2, ensure_ascii=False))

    print("\n3. Claude Desktop 再起動:")
    print("   設定ファイル変更後は Claude Desktop を完全に再起動")

    print("\n4. ログ確認:")
    print("   log/mcp_server.log でエラー詳細を確認")


def main():
    """メイン診断関数"""
    print("🩺 EPUB-LLM MCP Server 診断ツール")
    print("=" * 50)

    checks = [
        ("Python バージョン", check_python_version),
        ("依存関係", check_dependencies),
        ("プロジェクト構造", check_project_structure),
        ("ポート確認", check_port_availability),
        ("Claude 設定", check_claude_config),
        ("MCP Server テスト", test_mcp_server),
    ]

    all_passed = True
    for _name, check_func in checks:
        if not check_func():
            all_passed = False

    print("\n" + "=" * 50)

    if all_passed:
        print("🎉 全ての確認項目をパス！")
        print("   MCP Server は正常に動作するはずです。")
    else:
        print("⚠️ いくつかの問題が見つかりました")
        provide_solutions()


if __name__ == "__main__":
    main()
