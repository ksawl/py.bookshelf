# Bookshelf API (FastAPI) — краткая инструкция

Небольшой сервис для загрузки книг (docx/odt/pdf/txt) → разбиение на чанки → эмбеддинги → Pinecone (shared index, namespace = `book_{doc_id}`).

---

## Быстрый старт

```py
# Создание виртуального окружения и установка зависимостей
uv venv
source .venv/bin/activate  # Linux/Mac
# или
.venv\Scripts\activate     # Windows

uv pip install -e .        # Установка проекта в режиме разработки

# Запуск приложения
uv run main.py
```

## Переменные окружения (.env в корне проекта)

Перед запуском добавьте в корень .env файл с ключами для подключения

```env
PINECONE_API_KEY="your-pinecone-api-key"
OPENAI_API_KEY="your-openai-api-key"
```

Остальные константы проекта лежат по адресу ./app/core/config.py

---

## Основные эндпоинты

### 1) Загрузка книги

`POST /bookshelf`
multipart/form-data, поле `file`.

Пример curl:

```bash
curl -X POST "http://127.0.0.1:8000/bookshelf" \
  -F "file=@/path/to/book.docx" \
```

Ответ:

```json
{ "doc_id": "<uuid>", "status": "accepted" }
```

### 2) Статус задания

`GET /bookshelf/{booc_id}/status` — возвращает JobInfo (status, progress, total_chunks, index_name и т.д.)

### 3) Список загруженных книг

`GET /bookshelf` - возвращает список индексов загруженных книг из Pinecone

Пример curl:

```bash
curl -X GET "http://127.0.0.1:8000/bookshelf"
```

### 4) Получить метаинформацию о книге.

`GET /bookshelf/{book_id}` - возвращает метаинформацию о книге из Pinecone

Пример curl:

```bash
curl -X GET "http://127.0.0.1:8000/bookshelf/{book_id}"
```

### 5) Задать вопрос по книге.

`POST /bookshelf/{book_id}` - возвращает ответ и перечень ссылок на источник.

Пример curl:

```bash
curl -X POST "http://127.0.0.1:8000/bookshelf/{book_id}?q=<your-question>"
```

### 6) Удалить книгу и очистить связанный с ней индекс.

`DELETE /bookshelf/{book_id}`

Пример curl:

```bash
curl -X DELETE "http://127.0.0.1:8000/bookshelf/{book_id}"
```

Ответ:

```json
{"status": "accepted", "book_id": book_id}
```

---

## Коротко про поведение

-   Файлы временно сохраняются → фоновой таск обрабатывает: конвертация, извлечение заголовков, токенизация (tiktoken), chunking, генерация эмбеддингов (OpenAI или локально) → upsert в Pinecone под `namespace=book_{doc_id}`.
-   Первый батч эмбеддингов используется для определения dimension; если существующий индекс несовместим (sparse или другая dim) — создаётся новый dense-индекс `books-shared-index-dense-<dim>`.
