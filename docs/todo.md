-   [x] 根拠を示すために RAG 文脈へ書籍名（タイトル）を含める
-   [x] テスト実施（pytest: 57 passed, 1 skipped）

実行メモ:
- `/chat` のコンテキスト整形で、各スニペットのラベルに書籍タイトルを表示（例: `[○○入門#12] ...`）。
- EPUB→Markdown 抽出と FAISS インデックス化は実行済み（books=41, chunks=188）。
