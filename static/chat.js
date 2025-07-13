// Markdownパーサーのフォールバック機能
function safeParseMarkdown(text) {
	if (typeof marked !== 'undefined' && marked.parse) {
		try {
			return marked.parse(text)
		} catch (e) {
			console.warn('marked.parseでエラーが発生しました:', e)
			return text.replace(/\n/g, '<br>')
		}
	} else {
		console.warn(
			'markedライブラリが利用できません。プレーンテキストで表示します。'
		)
		return text.replace(/\n/g, '<br>')
	}
}

const form = document.getElementById('chat-form')
const messages = document.getElementById('messages')
const bookshelf = document.getElementById('bookshelf')
const bookshelfContainer = document.getElementById('bookshelf-container')
const bookshelfToggle = document.getElementById('bookshelf-toggle')
const noBooks = document.getElementById('no-books')
const statusArea = document.getElementById('status-area')
const selectedBooksList = document.getElementById('selected-books-list')

// ステータスメッセージ管理
let statusMessages = []
const MAX_STATUS_LINES = 20 // 表示する最大行数
const historyList = document.getElementById('history-list')
const bookshelfSelectAll = document.getElementById('bookshelf-select-all')

let selectedBooks = new Set()
let chatHistory = []
let timerInterval = null
let booksData = []
let currentSessionId = null

// 本棚取得
fetch('/bookshelf')
	.then((r) => {
		if (!r.ok) {
			throw new Error(`HTTP ${r.status}: ${r.statusText}`)
		}
		return r.json()
	})
	.then((books) => {
		booksData = books
		bookshelf.innerHTML = ''

		// 保存された書籍選択状態を復元、なければ全選択
		if (!loadSelectedBooksFromStorage()) {
			selectedBooks = new Set(books.map((b) => b.id)) // 初期状態で全選択
			updateStatus(
				`📚 初期状態で全書籍を選択しました (${books.length}冊)`
			)
		}

		if (books.length === 0) {
			bookshelf.style.display = 'none'
			noBooks.style.display = ''
		} else {
			bookshelf.style.display = ''
			noBooks.style.display = 'none'
			for (const book of books) {
				const div = document.createElement('div')
				div.className = 'bookshelf-item'
				div.tabIndex = 0
				div.dataset.bookId = book.id
				div.title = book.title
				const img = document.createElement('img')
				img.src = book.cover || ''
				img.alt = book.title
				img.className = 'bookshelf-cover'
				img.onerror = () => {
					img.style.display = 'none'
				}
				const caption = document.createElement('div')
				caption.className = 'bookshelf-title'
				caption.textContent = book.title
				caption.title = book.title // ツールチップで全文表示
				div.appendChild(img)
				div.appendChild(caption)
				if (selectedBooks.has(book.id)) {
					div.classList.add('selected') // 追加: 初期状態で選択状態
				}
				div.addEventListener('click', () => {
					if (div.classList.contains('selected')) {
						div.classList.remove('selected')
						selectedBooks.delete(book.id)
						updateStatus(`📚 「${book.title}」の選択を解除しました`)
					} else {
						div.classList.add('selected')
						selectedBooks.add(book.id)
						updateStatus(`📚 「${book.title}」を選択しました`)
					}
					updateSelectedBooksList()
					updateSelectAllButton() // 全選択ボタンの状態更新
					saveSelectedBooksToStorage() // 自動保存
				})
				bookshelf.appendChild(div)
			}
			updateSelectedBooksList()
		}
	})
	.catch((error) => {
		console.error('本棚の読み込みに失敗:', error)
		bookshelf.style.display = 'none'
		noBooks.style.display = ''
		noBooks.textContent = 'ネットワークエラー: 本棚の読み込みに失敗しました'

		// エラー時でも基本機能は使えるようにする
		booksData = []
		selectedBooks = new Set()
	})

function updateSelectAllButton() {
	if (!bookshelfSelectAll) return
	const allSelected =
		booksData.length > 0 &&
		booksData.every((book) => selectedBooks.has(book.id))
	if (allSelected) {
		bookshelfSelectAll.textContent = '全解除'
		bookshelfSelectAll.title = '全解除'
	} else {
		bookshelfSelectAll.textContent = '全選択'
		bookshelfSelectAll.title = '全選択'
	}
}

function updateSelectedBooksList() {
	if (!selectedBooksList) return
	const selectedBooksArr = booksData.filter((b) => selectedBooks.has(b.id))
	if (selectedBooksArr.length > 0) {
		selectedBooksList.innerHTML = ''
		selectedBooksArr.forEach((b) => {
			const chip = document.createElement('span')
			chip.className = 'selected-book-chip'
			chip.textContent = b.title
			const removeBtn = document.createElement('button')
			removeBtn.className = 'remove-book-btn'
			removeBtn.type = 'button'
			removeBtn.innerHTML = '&times;'
			removeBtn.title = 'この書籍を選択解除'
			removeBtn.addEventListener('click', (e) => {
				e.stopPropagation()
				selectedBooks.delete(b.id)
				const bookshelfItem = bookshelf.querySelector(
					`[data-book-id="${b.id}"]`
				)
				if (bookshelfItem) bookshelfItem.classList.remove('selected')
				updateSelectedBooksList()
				updateSelectAllButton() // 全選択ボタンの状態更新
				saveSelectedBooksToStorage() // 自動保存
			})
			chip.appendChild(removeBtn)
			selectedBooksList.appendChild(chip)
		})
		selectedBooksList.style.display = ''
	} else {
		selectedBooksList.innerHTML = ''
		selectedBooksList.style.display = 'none'
	}
	updateSelectAllButton()
}

// ステータスメッセージ更新関数
function updateStatus(message) {
	if (!message) return

	// 現在時刻を追加
	const timestamp = new Date().toLocaleTimeString()
	const messageWithTime = `[${timestamp}] ${message}`

	// メッセージを追加
	statusMessages.push(messageWithTime)

	// 最大行数を超えた場合、古いメッセージを削除
	if (statusMessages.length > MAX_STATUS_LINES) {
		statusMessages = statusMessages.slice(-MAX_STATUS_LINES)
	}

	// ステータスエリアを更新
	statusArea.innerHTML = statusMessages.join('<br>')

	// 自動スクロール（最新メッセージを表示）
	statusArea.scrollTop = statusArea.scrollHeight
}

// ステータスをクリア
function clearStatus() {
	statusMessages = []
	statusArea.innerHTML = ''
}

// 書籍選択状態をローカルストレージに保存
function saveSelectedBooksToStorage() {
	try {
		const selectedBooksArray = Array.from(selectedBooks)
		localStorage.setItem(
			'epub-llm-selected-books',
			JSON.stringify(selectedBooksArray)
		)
		// サイレント保存（メッセージ表示なし）
	} catch (error) {
		console.error('書籍選択状態の保存に失敗:', error)
		updateStatus('❌ 書籍選択状態の保存に失敗しました')
	}
}

// 書籍選択状態をローカルストレージから復元
function loadSelectedBooksFromStorage() {
	try {
		const saved = localStorage.getItem('epub-llm-selected-books')
		if (saved) {
			const selectedBooksArray = JSON.parse(saved)
			selectedBooks = new Set(selectedBooksArray)
			updateSelectedBooksList()
			updateSelectAllButton()

			// 本棚の選択状態も更新
			const bookshelfItems = document.querySelectorAll('.bookshelf-item')
			bookshelfItems.forEach((item) => {
				const bookId = item.dataset.bookId
				if (selectedBooks.has(bookId)) {
					item.classList.add('selected')
				} else {
					item.classList.remove('selected')
				}
			})

			updateStatus(
				`📚 保存された書籍選択を復元しました (${selectedBooksArray.length}冊)`
			)
			return true
		}
	} catch (error) {
		console.error('書籍選択状態の復元に失敗:', error)
		updateStatus('⚠️ 書籍選択状態の復元に失敗しました')
	}
	return false
}

// 書籍選択状態をクリア
function clearSelectedBooksStorage() {
	try {
		localStorage.removeItem('epub-llm-selected-books')
		selectedBooks.clear()
		updateSelectedBooksList()
		updateSelectAllButton()

		// 本棚の選択状態もクリア
		const bookshelfItems = document.querySelectorAll('.bookshelf-item')
		bookshelfItems.forEach((item) => {
			item.classList.remove('selected')
		})

		updateStatus('🗑️ 書籍選択状態をクリアしました')
	} catch (error) {
		console.error('書籍選択状態のクリアに失敗:', error)
		updateStatus('❌ 書籍選択状態のクリアに失敗しました')
	}
}

form.addEventListener('submit', async (e) => {
	e.preventDefault()
	const startTime = Date.now()
	const question = document.getElementById('question').value
	const selected = Array.from(selectedBooks)

	// ステータスをクリア
	clearStatus()

	// 書籍選択状況をステータスに表示
	if (selected.length > 0) {
		updateStatus(`📚 選択中の書籍: ${selected.length}冊`)
	} else {
		updateStatus('📚 書籍なしで質問中')
	}

	// ユーザーメッセージを表示
	const userDiv = document.createElement('div')
	userDiv.className = 'user-msg'
	userDiv.textContent = question
	messages.appendChild(userDiv)
	messages.scrollTop = messages.scrollHeight
	chatHistory.push({ role: 'user', content: question })
	document.getElementById('question').value = ''

	// AIメッセージの準備
	const aiDiv = document.createElement('div')
	aiDiv.className = 'ai-msg'
	aiDiv.textContent = '応答を準備中...'
	messages.appendChild(aiDiv)
	messages.scrollTop = messages.scrollHeight

	try {
		// 10分のタイムアウト設定
		const controller = new AbortController()
		const timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000) // 10分

		const response = await fetch('/chat', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ book_ids: selected, messages: chatHistory }),
			signal: controller.signal,
		})

		clearTimeout(timeoutId)

		if (!response.ok) {
			throw new Error(
				`サーバーエラー: ${response.status} ${response.statusText}`
			)
		}

		if (!response.body) {
			throw new Error('レスポンスボディが空です')
		}

		let aiMsg = ''
		aiDiv.textContent = ''
		const reader = response.body.getReader()

		try {
			while (true) {
				const { done, value } = await reader.read()
				if (done) break

				const chunk = new TextDecoder().decode(value)
				let obj
				try {
					obj = JSON.parse(chunk)
				} catch (parseError) {
					console.warn(
						'JSONパースエラー:',
						parseError,
						'chunk:',
						chunk
					)
					continue
				}

				if (obj.status) {
					updateStatus(obj.status)
				}

				if (obj.content) {
					// Remove unwanted template tokens and "Assistant" prefix
					let cleanContent = obj.content
						.replace(/assistant<\|header_end\|>/g, '')
						.replace(
							/<\|start_header_id\|>assistant<\|end_header_id\|>/g,
							''
						)
						.replace(/<\|.*?\|>/g, '')
						.replace(/^Assistant\s*:?\s*/i, '') // "Assistant:"を除去
					aiMsg += cleanContent
					aiDiv.innerHTML = safeParseMarkdown(aiMsg)
					aiDiv.scrollIntoView({ behavior: 'smooth', block: 'end' })
					messages.scrollTop = messages.scrollHeight
				}
			}
		} catch (readerError) {
			console.error('レスポンス読み取りエラー:', readerError)
			throw new Error('レスポンスの読み取り中にエラーが発生しました')
		}

		if (aiMsg.trim() === '') {
			aiMsg = '申し訳ありませんが、応答を生成できませんでした。'
			aiDiv.innerHTML = safeParseMarkdown(aiMsg)
		}

		// 最終的に "Assistant:" プレフィックスを除去
		aiMsg = aiMsg.replace(/^Assistant\s*:?\s*/i, '')
		aiDiv.innerHTML = safeParseMarkdown(aiMsg)

		chatHistory.push({ role: 'assistant', content: aiMsg })

		// 完了をステータスに表示
		const endTime = Date.now()
		const elapsed = ((endTime - startTime) / 1000).toFixed(2)
		updateStatus(`✅ 応答完了 (${elapsed}秒)`)

		// 履歴を自動保存
		try {
			await saveCurrentHistory()
		} catch (saveError) {
			console.error('履歴保存エラー:', saveError)
			updateStatus(`⚠️ 履歴保存に失敗: ${saveError.message}`)
			// 履歴保存に失敗してもチャットは続行
		}

		const timeDiv = document.createElement('div')
		timeDiv.className = 'loading-time'
		timeDiv.textContent = `応答時間: ${elapsed}秒`
		messages.appendChild(timeDiv)
		messages.scrollTop = messages.scrollHeight
	} catch (error) {
		console.error('チャット処理エラー:', error)

		// エラーメッセージを表示
		let errorMessage = 'エラーが発生しました。'
		if (
			error.message.includes('Failed to fetch') ||
			error.message.includes('Load failed')
		) {
			errorMessage =
				'ネットワーク接続エラーが発生しました。インターネット接続を確認してください。'
		} else if (error.message.includes('サーバーエラー')) {
			errorMessage = error.message
		} else {
			errorMessage = `エラー: ${error.message}`
		}

		// エラーをステータスエリアにも表示
		updateStatus(`❌ ${errorMessage}`)

		aiDiv.className = 'ai-msg error-msg'
		aiDiv.innerHTML = `<span style="color: #ff6b6b;">${errorMessage}</span>`

		// エラー時も履歴に記録（ユーザーメッセージのみ保持）
		chatHistory.pop() // ユーザーメッセージを一旦削除
		chatHistory.push({ role: 'user', content: question })

		const endTime = Date.now()
		const elapsed = ((endTime - startTime) / 1000).toFixed(2)
		const timeDiv = document.createElement('div')
		timeDiv.className = 'loading-time error'
		timeDiv.textContent = `エラー発生: ${elapsed}秒後`
		messages.appendChild(timeDiv)
		messages.scrollTop = messages.scrollHeight
	}
})

// テキストエリアで cmd+Enter（Mac）または ctrl+Enter で送信できるようにJSを追加
const questionTextarea = document.getElementById('question')
if (questionTextarea) {
	questionTextarea.addEventListener('keydown', (e) => {
		if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
			e.preventDefault()
			form.requestSubmit()
		}
	})
}

// 書籍選択UIをポップアップ化
bookshelfToggle.addEventListener('click', () => {
	bookshelfContainer.classList.toggle('open')
	if (bookshelfContainer.classList.contains('open')) {
		bookshelfContainer.style.position = 'fixed'
		bookshelfContainer.style.top = '50%'
		bookshelfContainer.style.left = '50%'
		bookshelfContainer.style.transform = 'translate(-50%, -50%)'
		bookshelfContainer.style.zIndex = '1000'
		bookshelfContainer.style.boxShadow = '0 8px 32px rgba(0,0,0,0.25)'
		bookshelfContainer.style.background = '#22303c'
		bookshelfContainer.style.maxHeight = '80vh'
		bookshelfContainer.style.overflowY = 'auto'
	} else {
		bookshelfContainer.style.position = ''
		bookshelfContainer.style.top = ''
		bookshelfContainer.style.left = ''
		bookshelfContainer.style.transform = ''
		bookshelfContainer.style.zIndex = ''
		bookshelfContainer.style.boxShadow = ''
		bookshelfContainer.style.background = ''
		bookshelfContainer.style.maxHeight = ''
		bookshelfContainer.style.overflowY = ''
	}
})

document.addEventListener('mousedown', (e) => {
	if (
		bookshelfContainer.classList.contains('open') &&
		!bookshelfContainer.contains(e.target) &&
		e.target !== bookshelfToggle
	) {
		bookshelfContainer.classList.remove('open')
		bookshelfContainer.style.position = ''
		bookshelfContainer.style.top = ''
		bookshelfContainer.style.left = ''
		bookshelfContainer.style.transform = ''
		bookshelfContainer.style.zIndex = ''
		bookshelfContainer.style.boxShadow = ''
		bookshelfContainer.style.background = ''
		bookshelfContainer.style.maxHeight = ''
		bookshelfContainer.style.overflowY = ''
	}
})

const bookshelfClose = document.getElementById('bookshelf-close')
if (bookshelfClose) {
	bookshelfClose.addEventListener('click', () => {
		bookshelfContainer.classList.remove('open')
		bookshelfContainer.style.position = ''
		bookshelfContainer.style.top = ''
		bookshelfContainer.style.left = ''
		bookshelfContainer.style.transform = ''
		bookshelfContainer.style.zIndex = ''
		bookshelfContainer.style.boxShadow = ''
		bookshelfContainer.style.background = ''
		bookshelfContainer.style.maxHeight = ''
		bookshelfContainer.style.overflowY = ''
	})
}

// 履歴サイドバーの取得・表示
async function loadHistoryList() {
	try {
		const res = await fetch('/list_histories')
		const summaries = await res.json()
		historyList.innerHTML = ''

		if (summaries.length === 0) {
			const emptyDiv = document.createElement('div')
			emptyDiv.className = 'history-empty'
			emptyDiv.textContent = '履歴がありません'
			historyList.appendChild(emptyDiv)
			return
		}

		summaries.forEach((summary) => {
			const li = document.createElement('li')
			li.className = 'history-item'
			li.dataset.sessionId = summary.session_id

			const titleDiv = document.createElement('div')
			titleDiv.className = 'history-title'
			titleDiv.textContent = summary.first_message || '新しいチャット'

			const metaDiv = document.createElement('div')
			metaDiv.className = 'history-meta'
			metaDiv.textContent = `${summary.message_count}件のメッセージ`
			if (summary.last_updated) {
				const date = new Date(summary.last_updated)
				metaDiv.textContent += ` • ${date.toLocaleDateString()}`
			}

			const deleteBtn = document.createElement('button')
			deleteBtn.className = 'history-delete-btn'
			deleteBtn.textContent = '削除'
			deleteBtn.title = '履歴を削除'
			deleteBtn.onclick = (e) => {
				e.stopPropagation()
				deleteHistory(summary.session_id)
			}

			li.appendChild(titleDiv)
			li.appendChild(metaDiv)
			li.appendChild(deleteBtn)

			li.addEventListener('click', () => {
				selectHistory(summary.session_id)
			})

			historyList.appendChild(li)
		})
	} catch (error) {
		console.error('履歴の読み込みに失敗:', error)
		historyList.innerHTML =
			'<div class="history-error">履歴の読み込みに失敗しました</div>'
	}
}

async function selectHistory(sessionId) {
	try {
		const res = await fetch(`/session/${sessionId}`)
		const result = await res.json()

		if (res.ok && result.messages && Array.isArray(result.messages)) {
			messages.innerHTML = ''
			chatHistory = []
			result.messages.forEach((msg) => {
				const div = document.createElement('div')
				if (msg.role === 'user') {
					div.className = 'user-msg'
					div.textContent = msg.content
				} else {
					div.className = 'ai-msg'
					div.innerHTML = safeParseMarkdown(msg.content)
				}
				messages.appendChild(div)
				chatHistory.push(msg)
			})
			currentSessionId = sessionId

			// 書籍選択状態を復元
			if (result.book_ids && Array.isArray(result.book_ids)) {
				selectedBooks = new Set(result.book_ids)
				updateSelectedBooksList()
				updateStatus(`📚 書籍選択を復元: ${result.book_ids.length}冊`)

				// 本棚の選択状態も更新
				const bookshelfItems =
					document.querySelectorAll('.bookshelf-item')
				bookshelfItems.forEach((item) => {
					const bookId = item.dataset.bookId
					if (selectedBooks.has(bookId)) {
						item.classList.add('selected')
					} else {
						item.classList.remove('selected')
					}
				})
			}

			// 選択状態を更新
			Array.from(historyList.children).forEach((li) => {
				li.classList.toggle(
					'selected',
					li.dataset.sessionId === sessionId
				)
			})

			// メッセージ領域をスクロール
			messages.scrollTop = messages.scrollHeight
		} else {
			console.error('履歴の読み込みに失敗:', result.error)
		}
	} catch (error) {
		console.error('履歴の読み込みエラー:', error)
	}
}

async function deleteHistory(sessionId) {
	if (!confirm('この履歴を削除しますか？')) {
		return
	}

	try {
		const res = await fetch(`/history/${sessionId}`, {
			method: 'DELETE',
		})
		const result = await res.json()

		if (res.ok) {
			// 削除成功時の処理
			if (currentSessionId === sessionId) {
				// 現在表示中の履歴を削除した場合、画面をクリア
				messages.innerHTML = ''
				chatHistory = []
				currentSessionId = null
			}
			// 履歴リストを再読み込み
			await loadHistoryList()
		} else {
			console.error('履歴の削除に失敗:', result.error)
			alert('履歴の削除に失敗しました')
		}
	} catch (error) {
		console.error('履歴の削除エラー:', error)
		alert('履歴の削除中にエラーが発生しました')
	}
}

async function saveCurrentHistory() {
	if (chatHistory.length === 0) return

	// セッションIDが未設定の場合、新しいIDを生成
	if (!currentSessionId) {
		currentSessionId = `session_${Date.now()}_${Math.random()
			.toString(36)
			.substr(2, 9)}`
	}

	try {
		const saveData = {
			messages: chatHistory,
			book_ids: Array.from(selectedBooks),
		}
		const res = await fetch(`/history/${currentSessionId}`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(saveData),
		})

		if (res.ok) {
			console.log('履歴を保存しました:', currentSessionId)
			// 履歴リストを更新（ただし新しい会話の場合のみ）
			const isNewSession = !Array.from(historyList.children).some(
				(li) => li.dataset.sessionId === currentSessionId
			)
			if (isNewSession) {
				await loadHistoryList()
			}
		} else {
			const errorText = await res.text()
			console.error('履歴の保存に失敗しました:', res.status, errorText)
		}
	} catch (error) {
		console.error('履歴保存エラー:', error)
	}
}

function startNewChat() {
	messages.innerHTML = ''
	chatHistory = []
	currentSessionId = null
	clearStatus() // ステータスもクリア

	// 履歴リストの選択状態をクリア
	Array.from(historyList.children).forEach((li) => {
		li.classList.remove('selected')
	})

	// 書籍選択は維持する（ユーザーが明示的にクリアしない限り）
	// 必要に応じて書籍選択もクリアしたい場合は、以下のコメントを外す
	// selectedBooks.clear()
	// updateSelectedBooksList()
}

// 新しいチャットボタンを追加（HTMLに対応するボタンがあると想定）
const newChatBtn = document.getElementById('new-chat-btn')
if (newChatBtn) {
	newChatBtn.addEventListener('click', startNewChat)
}

window.addEventListener('DOMContentLoaded', () => {
	loadHistoryList()
	// 本棚データ取得時に既に復元済みなので、ここでは呼ばない
})

if (bookshelfSelectAll) {
	bookshelfSelectAll.addEventListener('click', () => {
		const allSelected =
			booksData.length > 0 &&
			booksData.every((book) => selectedBooks.has(book.id))
		if (allSelected) {
			// 全解除
			selectedBooks.clear()
			Array.from(
				bookshelf.querySelectorAll('.bookshelf-item.selected')
			).forEach((item) => item.classList.remove('selected'))
			bookshelfSelectAll.textContent = '全選択'
			bookshelfSelectAll.title = '全選択'
			updateStatus('📚 書籍選択をすべて解除しました')
		} else {
			// 全選択
			booksData.forEach((book) => {
				selectedBooks.add(book.id)
				const bookshelfItem = bookshelf.querySelector(
					`[data-book-id="${book.id}"]`
				)
				if (bookshelfItem) bookshelfItem.classList.add('selected')
			})
			bookshelfSelectAll.textContent = '全解除'
			bookshelfSelectAll.title = '全解除'
			updateStatus(
				`📚 すべての書籍を選択しました (${booksData.length}冊)`
			)
		}
		updateSelectedBooksList()
		saveSelectedBooksToStorage() // 自動保存
	})
}

// 全解除ボタンの機能を追加
const bookshelfClearAll = document.getElementById('bookshelf-clear-all')
if (bookshelfClearAll) {
	bookshelfClearAll.addEventListener('click', () => {
		// 全解除
		selectedBooks.clear()
		Array.from(
			bookshelf.querySelectorAll('.bookshelf-item.selected')
		).forEach((item) => item.classList.remove('selected'))
		updateSelectedBooksList()
		updateSelectAllButton()
		updateStatus('📚 書籍選択をすべて解除しました')
		saveSelectedBooksToStorage() // 自動保存
	})
}
