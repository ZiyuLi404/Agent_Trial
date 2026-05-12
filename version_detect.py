import requests
import hashlib
import os
from bs4 import BeautifulSoup
from datetime import datetime


print("正在运行：方案1 搜索结果检测版本")


SEARCH_TARGETS = {
    "chatgpt_release_notes": "ChatGPT release notes OpenAI",
    "model_release_notes": "OpenAI model release notes",
    "chatgpt_agent_release_notes": "ChatGPT agent release notes OpenAI"
}

HASH_DIR = "hash_records"
LOG_FILE = "update_log.txt"


def search_bing(query):
    """
    用 Bing 搜索公开信息，而不是直接访问 help.openai.com。
    """
    url = "https://www.bing.com/search"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    params = {
        "q": query
    }

    response = requests.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()

    return response.text


def extract_search_text(html):
    """
    从搜索结果页面提取文本内容。
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    # 只取前 120 行，减少广告、推荐内容带来的干扰
    return "\n".join(lines[:120])


def calculate_hash(text):
    """
    计算文本 hash。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_hash_file(name):
    if not os.path.exists(HASH_DIR):
        os.makedirs(HASH_DIR)

    return os.path.join(HASH_DIR, f"{name}.txt")


def read_old_hash(name):
    file_path = get_hash_file(name)

    if not os.path.exists(file_path):
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def save_new_hash(name, new_hash):
    file_path = get_hash_file(name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_hash)


def write_log(message):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def check_one_target(name, query):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        html = search_bing(query)
        text = extract_search_text(html)

        new_hash = calculate_hash(text)
        old_hash = read_old_hash(name)

        if old_hash is None:
            save_new_hash(name, new_hash)
            message = f"[{now}] {name}: 第一次运行，已保存当前搜索结果状态。"
            print(message)
            write_log(message)
            return False

        if new_hash != old_hash:
            save_new_hash(name, new_hash)
            message = f"[{now}] {name}: 可能有更新，搜索结果发生变化。"
            print(message)
            write_log(message)
            return True

        message = f"[{now}] {name}: 没有检测到变化。"
        print(message)
        write_log(message)
        return False

    except Exception as e:
        message = f"[{now}] {name}: 检测失败，原因：{e}"
        print(message)
        write_log(message)
        return False


def check_all_updates():
    updated_items = []

    for name, query in SEARCH_TARGETS.items():
        updated = check_one_target(name, query)

        if updated:
            updated_items.append(name)

    if updated_items:
        print("\n本次检测发现以下项目可能有更新：")
        for item in updated_items:
            print("-", item)
    else:
        print("\n本次检测没有发现明显更新。")


if __name__ == "__main__":
    check_all_updates()