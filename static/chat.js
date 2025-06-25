const form = document.getElementById('chat-form')
const messages = document.getElementById('messages')
const bookshelf = document.getElementById('bookshelf')
const bookshelfContainer = document.getElementById('bookshelf-container')
const bookshelfToggle = document.getElementById('bookshelf-toggle')
const noBooks = document.getElementById('no-books')
const statusArea = document.getElementById('status-area')
const selectedBooksList = document.getElementById('selected-books-list')
const historyList = document.getElementById('history-list')
const bookshelfSelectAll = document.getElementById('bookshelf-select-all')

let selectedBooks = new Set()
let chatHistory = []
let timerInterval = null
let booksData = []
let currentSessionId = null

// 本棚取得
fetch('/bookshelf')
	.then((r) => r.json())
	.then((books) => {
		booksData = books
		bookshelf.innerHTML = ''
		selectedBooks = new Set(books.map((b) => b.id)) // 追加: 初期状態で全選択
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
				div.appendChild(img)
				div.appendChild(caption)
				if (selectedBooks.has(book.id)) {
					div.classList.add('selected') // 追加: 初期状態で選択状態
				}
				div.addEventListener('click', () => {
					if (div.classList.contains('selected')) {
						div.classList.remove('selected')
						selectedBooks.delete(book.id)
					} else {
						div.classList.add('selected')
						selectedBooks.add(book.id)
					}
					updateSelectedBooksList()
				})
				bookshelf.appendChild(div)
			}
			updateSelectedBooksList()
		}
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

async function playVoiceVox(text) {
	try {
		const res1 = await fetch(
			'http://localhost:50021/audio_query?speaker=1',
			{
				method: 'POST',
				headers: {
					'Content-Type': 'application/x-www-form-urlencoded',
				},
				body: `text=${encodeURIComponent(text)}`,
			}
		)
		if (!res1.ok) return
		const query = await res1.json()
		const res2 = await fetch('http://localhost:50021/synthesis?speaker=1', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(query),
		})
		if (!res2.ok) return
		const blob = await res2.blob()
		const url = URL.createObjectURL(blob)
		const audio = new Audio(url)
		audio.play()
	} catch (e) {
		console.error('VOICEVOX連携エラー', e)
	}
}

form.addEventListener('submit', async (e) => {
	e.preventDefault()
	const startTime = Date.now()
	const question = document.getElementById('question').value
	const selected = Array.from(selectedBooks)
	const userDiv = document.createElement('div')
	userDiv.className = 'user-msg'
	userDiv.textContent = question
	messages.appendChild(userDiv)
	// ChatGPT風: ユーザー送信時に自動スクロール
	messages.scrollTop = messages.scrollHeight
	chatHistory.push({ role: 'user', content: question })
	document.getElementById('question').value = ''
	const aiDiv = document.createElement('div')
	aiDiv.className = 'ai-msg'
	messages.appendChild(aiDiv)
	// ChatGPT風: AI応答時も自動スクロール
	messages.scrollTop = messages.scrollHeight
	const response = await fetch('/chat', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ book_ids: selected, messages: chatHistory }),
	})
	if (!response.ok || !response.body) {
		aiDiv.textContent = `サーバーエラー: ${response.status}`
		return
	}
	let aiMsg = ''
	aiDiv.textContent = ''
	const reader = response.body.getReader()
	while (true) {
		const { done, value } = await reader.read()
		if (done) break
		const chunk = new TextDecoder().decode(value)
		let obj
		try {
			obj = JSON.parse(chunk)
		} catch (e) {
			continue
		}
		if (obj.status) {
			statusArea.innerHTML = marked.parse(obj.status)
			statusArea.scrollTop = statusArea.scrollHeight
		}
		if (obj.content) {
			aiMsg += obj.content
			console.log('Received chunk:', obj.content)
			aiDiv.innerHTML = marked.parse(aiMsg)
			aiDiv.scrollIntoView({ behavior: 'smooth', block: 'end' })
			// ChatGPT風: AI応答時も自動スクロール
			messages.scrollTop = messages.scrollHeight
		}
	}
	chatHistory.push({ role: 'assistant', content: aiMsg })
	const endTime = Date.now()
	const elapsed = ((endTime - startTime) / 1000).toFixed(2)
	const timeDiv = document.createElement('div')
	timeDiv.className = 'loading-time'
	timeDiv.textContent = `応答時間: ${elapsed}秒`
	messages.appendChild(timeDiv)
	// 応答後も自動スクロール
	messages.scrollTop = messages.scrollHeight
	if (aiMsg) playVoiceVox(aiMsg)
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
	const res = await fetch('/history_list')
	const list = await res.json()
	historyList.innerHTML = ''
	list.forEach((id) => {
		const li = document.createElement('li')
		li.textContent = id
		li.addEventListener('click', () => {
			selectHistory(id)
		})
		historyList.appendChild(li)
	})
}

async function selectHistory(sessionId) {
	const res = await fetch(`/history/${sessionId}`)
	const history = await res.json()
	if (Array.isArray(history)) {
		messages.innerHTML = ''
		chatHistory = []
		history.forEach((msg) => {
			const div = document.createElement('div')
			if (msg.role === 'user') {
				div.className = 'user-msg'
			} else {
				div.className = 'ai-msg'
			}
			div.textContent = msg.content
			messages.appendChild(div)
			chatHistory.push(msg)
		})
		currentSessionId = sessionId
		// 選択状態を更新
		Array.from(historyList.children).forEach((li) => {
			li.classList.toggle('selected', li.textContent === sessionId)
		})
	}
}

window.addEventListener('DOMContentLoaded', () => {
	loadHistoryList()
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
		}
		updateSelectedBooksList()
	})
}
