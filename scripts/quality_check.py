#!/usr/bin/env python3
"""
コード品質チェックツール

black, pylint, mypy, ruffを実行してコード品質をチェックします。
t-wada流TDD開発プロセスに従い、品質基準を厳格に管理します。
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# from typing import Any  # Removed unused import


class QualityChecker:
    """コード品質チェッカー"""

    def __init__(self, target_dir: str = "src"):
        self.target_dir = target_dir
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

    def setup_logging(self) -> None:
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    def run_command(self, cmd: list[str], description: str) -> tuple[bool, str]:
        """コマンドを実行して結果を返す"""
        self.logger.info("🔍 %s を実行中...", description)
        try:
            result = subprocess.run(
                cmd,
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                self.logger.info("✅ %s: 成功", description)
                return True, result.stdout
            self.logger.error("❌ %s: 失敗", description)
            self.logger.error("stdout: %s", result.stdout)
            self.logger.error("stderr: %s", result.stderr)
            return False, result.stderr
        except FileNotFoundError:
            self.logger.error("❌ %s: コマンドが見つかりません", description)
            return False, f"Command not found: {cmd[0]}"
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("❌ %s: 実行エラー: %s", description, e)
            return False, str(e)

    def check_ruff_format(self) -> bool:
        """ruffでコードフォーマットをチェック"""
        success, _ = self.run_command(
            ["ruff", "format", "--check", self.target_dir],
            "ruff format (フォーマットチェック)",
        )
        return success

    def fix_ruff_format(self) -> bool:
        """ruffでコードフォーマットを修正"""
        success, _ = self.run_command(
            ["ruff", "format", self.target_dir], "ruff format (フォーマット修正)"
        )
        return success

    def check_ruff_lint(self) -> bool:
        """ruffでリンティングをチェック"""
        success, _ = self.run_command(
            ["ruff", "check", self.target_dir], "ruff lint (リンティングチェック)"
        )
        return success

    def fix_ruff_lint(self) -> bool:
        """ruffでリンティング問題を修正"""
        success, _ = self.run_command(
            ["ruff", "check", "--fix", self.target_dir], "ruff lint (リンティング修正)"
        )
        return success

    def check_black(self) -> bool:
        """blackでフォーマットをチェック"""
        success, _ = self.run_command(
            ["black", "--check", "--diff", self.target_dir],
            "black (フォーマットチェック)",
        )
        return success

    def fix_black(self) -> bool:
        """blackでフォーマットを修正"""
        success, _ = self.run_command(
            ["black", self.target_dir], "black (フォーマット修正)"
        )
        return success

    def check_mypy(self) -> bool:
        """mypyで型チェック"""
        success, _ = self.run_command(["mypy", self.target_dir], "mypy (型チェック)")
        return success

    def check_pylint(self) -> bool:
        """pylintでコードスタイルチェック"""
        success, _ = self.run_command(
            ["pylint", self.target_dir], "pylint (コードスタイルチェック)"
        )
        return success

    def run_all_checks(self, fix: bool = False) -> bool:
        """すべてのチェックを実行"""
        self.logger.info("🚀 コード品質チェックを開始します...")
        self.logger.info("📁 対象ディレクトリ: %s", self.target_dir)

        if fix:
            self.logger.info("🔧 修正モード: 自動修正を実行します")
        else:
            self.logger.info("🔍 チェックモード: 問題を報告します")

        results = []

        # 1. ruff format
        if fix:
            results.append(("ruff format", self.fix_ruff_format()))
        else:
            results.append(("ruff format", self.check_ruff_format()))

        # 2. ruff lint
        if fix:
            results.append(("ruff lint", self.fix_ruff_lint()))
        else:
            results.append(("ruff lint", self.check_ruff_lint()))

        # 3. black (修正モードでのみ実行、チェックはruffで代用)
        if fix:
            results.append(("black", self.fix_black()))

        # 4. mypy (チェックのみ)
        results.append(("mypy", self.check_mypy()))

        # 5. pylint (チェックのみ)
        results.append(("pylint", self.check_pylint()))

        # 結果サマリー
        self.logger.info("=" * 60)
        self.logger.info("📊 品質チェック結果サマリー")
        self.logger.info("=" * 60)

        success_count = 0
        for tool, success in results:
            status = "✅ 成功" if success else "❌ 失敗"
            self.logger.info("%-15s: %s", tool, status)
            if success:
                success_count += 1

        total = len(results)
        self.logger.info("-" * 60)
        self.logger.info(
            "🎯 結果: %d/%d 成功 (%.1f%%)",
            success_count,
            total,
            (success_count / total) * 100,
        )

        if success_count == total:
            self.logger.info("🎉 すべてのチェックが成功しました！")
            return True
        self.logger.warning("⚠️  一部のチェックが失敗しました")
        return False

    def run_specific_tool(self, tool: str, fix: bool = False) -> bool:
        """特定のツールのみ実行"""
        self.logger.info("🔧 %s を実行します", tool)

        if tool == "ruff-format":
            return self.fix_ruff_format() if fix else self.check_ruff_format()
        if tool == "ruff-lint":
            return self.fix_ruff_lint() if fix else self.check_ruff_lint()
        if tool == "black":
            return self.fix_black() if fix else self.check_black()
        if tool == "mypy":
            return self.check_mypy()
        if tool == "pylint":
            return self.check_pylint()
        self.logger.error("❌ 不明なツール: %s", tool)
        return False


def main() -> None:
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="コード品質チェックツール (t-wada流TDD準拠)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  %(prog)s                    # 全ツールでチェック
  %(prog)s --fix              # 全ツールでチェック＋修正
  %(prog)s --tool ruff-lint   # ruff lintのみ実行
  %(prog)s --tool mypy        # mypyのみ実行
  %(prog)s --dir test         # testディレクトリを対象

利用可能なツール:
  ruff-format  - ruffフォーマット
  ruff-lint    - ruffリンティング
  black        - blackフォーマット
  mypy         - 型チェック
  pylint       - コードスタイルチェック
""",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="可能な場合は問題を自動修正する",
    )

    parser.add_argument(
        "--tool",
        choices=["ruff-format", "ruff-lint", "black", "mypy", "pylint"],
        help="実行する特定のツールを指定",
    )

    parser.add_argument(
        "--dir",
        default="src",
        help="チェック対象のディレクトリ (デフォルト: src)",
    )

    args = parser.parse_args()

    checker = QualityChecker(args.dir)

    if args.tool:
        success = checker.run_specific_tool(args.tool, args.fix)
    else:
        success = checker.run_all_checks(args.fix)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
