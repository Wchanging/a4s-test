from openai import OpenAI
import os
import json
import dotenv
from typing import Dict, List, Any

# 加载环境变量
dotenv.load_dotenv()


def generate_user_profile(user_data: Dict[str, Any], api_key: str = None, model_name: str = "qwen-max",
                          use_multimodal: bool = False) -> Dict[str, Any]:
    """
    根据用户历史记录生成用户画像，支持纯文本或多模态分析

    Args:
        user_data: 单个用户的历史数据（包含uid和articles）
        api_key: API密钥，如果不提供则从环境变量获取
        model_name: 模型名称，默认qwen-max（纯文本），qwen-vl-max（多模态）
        use_multimodal: 是否使用多模态分析（图片+视频+文本）

    Returns:
        用户画像JSON格式数据
    """

    # 根据多模态设置选择合适的模型
    if use_multimodal and model_name == "qwen-max":
        model_name = "qwen-vl-max"
    elif not use_multimodal and model_name == "qwen-vl-max":
        model_name = "qwen-max"

    # 初始化客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    uid = user_data.get("uid", "")
    articles = user_data.get("articles", [])

    if not articles:
        return {
            "uid": uid,
            "error": "没有足够的历史数据"
        }

    # 根据模式构建不同的消息
    if use_multimodal:
        messages = build_multimodal_messages(uid, articles)
    else:
        messages = build_text_only_messages(uid, articles)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            # max_tokens=1000
        )

        result_text = response.choices[0].message.content.strip()

        # 尝试解析JSON
        try:
            # 提取JSON部分（去除可能的markdown标记）
            if "```json" in result_text:
                json_start = result_text.find("```json") + 7
                json_end = result_text.find("```", json_start)
                result_text = result_text[json_start:json_end]
            elif "```" in result_text:
                json_start = result_text.find("```") + 3
                json_end = result_text.rfind("```")
                result_text = result_text[json_start:json_end]

            profile = json.loads(result_text.strip())
            return profile

        except json.JSONDecodeError as e:
            return {
                "uid": uid,
                "error": f"JSON解析失败: {str(e)}",
                "raw_response": result_text
            }

    except Exception as e:
        return {
            "uid": uid,
            "error": f"API调用失败: {str(e)}"
        }


def build_content_summary(articles: List[Dict[str, Any]]) -> str:
    """
    构建用户内容摘要

    Args:
        articles: 用户的文章列表

    Returns:
        内容摘要字符串
    """
    summary_parts = []

    for i, article in enumerate(articles, 1):
        article_id = article.get("article_id", "")
        article_content = article.get("article_content", "")
        comments = article.get("comments", [])

        # 文章内容
        if article_content:
            summary_parts.append(f"文章{i} (ID: {article_id}):")
            summary_parts.append(f"内容: {article_content[:200]}...")

        # 评论内容
        if comments:
            summary_parts.append(f"该文章下的评论({len(comments)}条):")
            comment_count = 0
            for comment_group in comments:
                parent_comment = comment_group.get("parent_comment", "")
                content_list = comment_group.get("content", [])

                if parent_comment:
                    summary_parts.append(f"  回复「{parent_comment}」:")
                else:
                    summary_parts.append(f"  顶级评论:")

                for content_item in content_list[:3]:  # 只取前3条评论
                    comment_text = content_item.get("content", "")
                    if comment_text:
                        summary_parts.append(f"    - {comment_text}")
                        comment_count += 1
                        if comment_count >= 10:  # 限制评论数量
                            break

                if comment_count >= 10:
                    break

        summary_parts.append("")  # 文章间分隔

        if i >= 3:  # 只分析前3篇文章
            break

    return "\n".join(summary_parts)


def process_user_histories(file_path: str, output_path: str, api_key: str = None,
                           model_name: str = "qwen-max", use_multimodal: bool = False):
    """
    批量处理用户历史数据，生成用户画像

    Args:
        file_path: 用户历史数据文件路径
        output_path: 输出文件路径
        api_key: API密钥
        model_name: 模型名称
        use_multimodal: 是否使用多模态分析
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            user_histories = json.load(f)

        profiles = []
        mode_str = "多模态" if use_multimodal else "纯文本"
        print(f"开始{mode_str}用户画像分析，共{len(user_histories)}个用户")

        for i, user_data in enumerate(user_histories):
            print(f"正在处理用户 {i+1}/{len(user_histories)}: {user_data.get('uid', 'Unknown')} ({mode_str}模式)")

            profile = generate_user_profile(user_data, api_key, model_name, use_multimodal)
            profiles.append(profile)

            # 可以添加延时避免API限流
            # time.sleep(1)
            # 提前保存，避免长时间运行导致数据丢失
            if (i + 1) % 10 == 0:  # 每10个用户保存一次
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(profiles, f, ensure_ascii=False, indent=2)
                print(f"已处理{i+1}个用户，中间结果已保存")

        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)

        print(f"用户画像生成完成，结果保存到: {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")


def test_content_summary(user_data: Dict[str, Any]) -> str:
    """
    测试内容摘要生成功能（不调用API）

    Args:
        user_data: 用户数据

    Returns:
        内容摘要
    """
    uid = user_data.get("uid", "")
    articles = user_data.get("articles", [])

    print(f"用户ID: {uid}")
    print(f"文章数量: {len(articles)}")

    if not articles:
        return "没有文章数据"

    summary = build_content_summary(articles)
    return summary


def analyze_user_content_manually(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    手动分析用户内容（不使用API，用于测试）

    Args:
        user_data: 用户数据

    Returns:
        简单的用户画像分析
    """
    uid = user_data.get("uid", "")
    articles = user_data.get("articles", [])

    analysis = {
        "uid": uid,
        "article_count": len(articles),
        "total_comments": 0,
        "content_keywords": [],
        "sample_contents": [],
        "media_summary": {
            "total_images": 0,
            "total_videos": 0,
            "image_urls": [],
            "video_urls": [],
            "has_media_content": False
        }
    }

    all_text = []

    for article in articles:
        # 收集文章内容
        article_content = article.get("article_content", "")
        if article_content:
            all_text.append(article_content)
            analysis["sample_contents"].append(article_content[:100] + "...")

        # 统计文章媒体内容
        article_images = article.get("article_images", [])
        article_videos = article.get("article_videos", [])
        analysis["media_summary"]["total_images"] += len(article_images)
        analysis["media_summary"]["total_videos"] += len(article_videos)
        analysis["media_summary"]["image_urls"].extend(article_images[:2])  # 保存前2个URL作为样本
        analysis["media_summary"]["video_urls"].extend(article_videos[:2])

        if article_images or article_videos:
            analysis["media_summary"]["has_media_content"] = True

        # 收集评论内容
        comments = article.get("comments", [])
        for comment_group in comments:
            comment_contents = comment_group.get("content", [])
            analysis["total_comments"] += len(comment_contents)

            for content_item in comment_contents:
                comment_text = content_item.get("content", "")
                if comment_text:
                    all_text.append(comment_text)

                # 统计评论中的媒体
                comment_images = content_item.get("images", [])
                comment_videos = content_item.get("videos", [])
                analysis["media_summary"]["total_images"] += len(comment_images)
                analysis["media_summary"]["total_videos"] += len(comment_videos)
                analysis["media_summary"]["image_urls"].extend(comment_images[:1])
                analysis["media_summary"]["video_urls"].extend(comment_videos[:1])

                if comment_images or comment_videos:
                    analysis["media_summary"]["has_media_content"] = True

    # 简单的关键词提取
    full_text = " ".join(all_text)
    keywords = ["小米", "SU7", "智驾", "安全", "事故", "爆燃", "责任", "技术", "电池", "自动驾驶"]
    found_keywords = [kw for kw in keywords if kw in full_text]
    analysis["content_keywords"] = found_keywords

    return analysis


def build_multimodal_messages(uid: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    构建多模态消息，包含文本、图片和视频

    Args:
        uid: 用户ID
        articles: 用户文章列表

    Returns:
        消息列表
    """
    # 构建文本内容摘要
    text_summary = build_text_content_summary(articles)

    # 收集图片和视频URLs
    image_urls = collect_image_urls(articles)
    video_urls = collect_video_urls(articles)

    # 构建基础prompt
    text_prompt = f"""
请根据以下用户在小米SU7高速碰撞爆燃事件中的发言、转发内容和分享的图片/视频，分析该用户的画像特征。

用户ID: {uid}

用户文本内容:
{text_summary}

媒体内容统计:
- 图片数量: {len(image_urls)}
- 视频数量: {len(video_urls)}

请从以下维度分析用户画像，并以JSON格式输出：

1. 立场(stance): 对小米在此事件中的态度（支持小米/反对小米/中立/未明确）
2. 情感(emotion): 主要情感倾向（愤怒/同情/理性/冷漠/担忧/悲伤等，最多3个词）
3. 认知角度(perspective): 关注重点（技术安全/企业责任/用户教育/法律责任/行业发展/媒体传播等，最多3个）
4. 表达风格(style): 表达特点（客观理性/情绪化/专业分析/传播转发/质疑批判等，最多2个）
5. 参与程度(engagement): 参与热度（高度关注/一般关注/偶然提及）
6. 信息倾向(info_tendency): 信息来源偏好（官方信息/媒体报道/个人观点/专家分析/图片信息/视频内容）
7. 媒体使用(media_usage): 媒体内容特征（无媒体/新闻截图/现场图片/新闻视频/现场视频/图文并茂/视频为主等）

要求：
- 每个维度提供简洁的关键词，不要长句
- 结合文本内容、图片内容和视频内容进行综合分析
- 重点关注图片/视频中的文字信息、情感色彩、内容类型
- 基于实际内容分析，避免推测
- 如果某个维度信息不足，标注"信息不足"
- 严格按照JSON格式输出

输出格式：
{{
    "uid": "{uid}",
    "stance": "中立",
    "emotion": ["担忧", "理性"],
    "perspective": ["技术安全", "用户教育"],
    "style": ["客观理性"],
    "engagement": "高度关注",
    "info_tendency": "媒体报道",
    "media_usage": "新闻视频"
}}
"""

    # 构建消息内容
    content_parts = [{"type": "text", "text": text_prompt}]

    # 添加图片（最多3张避免token过多）
    for img_url in image_urls[:3]:
        if img_url and img_url.startswith("http"):
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })

    # 添加视频（最多2个避免token过多）
    for video_url in video_urls[:2]:
        if video_url and video_url.startswith("http"):
            content_parts.append({
                "type": "video_url",
                "video_url": {"url": video_url}
            })

    messages = [
        {
            "role": "user",
            "content": content_parts
        }
    ]

    return messages


def build_text_only_messages(uid: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    构建纯文本消息（不包含图片和视频）

    Args:
        uid: 用户ID
        articles: 用户文章列表

    Returns:
        消息列表
    """
    # 构建文本内容摘要
    text_summary = build_text_content_summary(articles)

    # 收集媒体统计信息（用于分析但不包含在API调用中）
    image_urls = collect_image_urls(articles)
    video_urls = collect_video_urls(articles)

    # 构建纯文本prompt
    text_prompt = f"""
请根据以下用户在小米SU7高速碰撞爆燃事件中的发言和转发内容，分析该用户的画像特征。

用户ID: {uid}

用户文本内容:
{text_summary}

媒体内容统计（仅统计数量，不分析具体内容）:
- 分享图片数量: {len(image_urls)}
- 分享视频数量: {len(video_urls)}

请从以下维度分析用户画像，并以JSON格式输出：

1. 立场(stance): 对小米在此事件中的态度（支持小米/反对小米/中立/未明确）
2. 情感(emotion): 主要情感倾向（愤怒/同情/理性/冷漠/担忧/悲伤等，最多3个词）
3. 认知角度(perspective): 关注重点（技术安全/企业责任/用户教育/法律责任/行业发展/媒体传播等，最多3个）
4. 表达风格(style): 表达特点（客观理性/情绪化/专业分析/传播转发/质疑批判等，最多2个）
5. 参与程度(engagement): 参与热度（高度关注/一般关注/偶然提及）
6. 信息倾向(info_tendency): 信息来源偏好（官方信息/媒体报道/个人观点/专家分析）
7. 媒体使用(media_usage): 基于数量推断的媒体使用特征（无媒体/偶尔配图/经常配图/视频分享/图文并茂等）

要求：
- 每个维度提供简洁的关键词，不要长句
- 基于文本内容和媒体使用频率进行分析
- 媒体使用维度主要基于数量统计，不涉及具体内容
- 基于实际内容分析，避免推测
- 如果某个维度信息不足，标注"信息不足"
- 严格按照JSON格式输出

输出格式：
{{
    "uid": "{uid}",
    "stance": "中立",
    "emotion": ["担忧", "理性"],
    "perspective": ["技术安全", "用户教育"],
    "style": ["客观理性"],
    "engagement": "高度关注",
    "info_tendency": "媒体报道",
    "media_usage": "偶尔配图"
}}
"""

    messages = [
        {
            "role": "user",
            "content": text_prompt
        }
    ]

    return messages


def build_text_content_summary(articles: List[Dict[str, Any]]) -> str:
    """
    构建纯文本内容摘要
    """
    summary_parts = []

    for i, article in enumerate(articles, 1):
        article_id = article.get("article_id", "")
        article_content = article.get("article_content", "")
        comments = article.get("comments", [])

        # 文章内容
        if article_content:
            summary_parts.append(f"文章{i}:")
            summary_parts.append(f"{article_content}")

        # 评论内容
        if comments:
            summary_parts.append(f"该文章下的评论:")
            comment_count = 0
            for comment_group in comments:
                parent_comment = comment_group.get("parent_comment", "")
                content_list = comment_group.get("content", [])

                for content_item in content_list[:3]:  # 只取前3条评论
                    comment_text = content_item.get("content", "")
                    if comment_text:
                        if parent_comment:
                            summary_parts.append(f"回复「{parent_comment}」: {comment_text}")
                        else:
                            summary_parts.append(f"评论: {comment_text}")
                        comment_count += 1
                        if comment_count >= 8:
                            break

                if comment_count >= 8:
                    break

        summary_parts.append("")  # 文章间分隔

        if i >= 2:  # 只分析前2篇文章
            break

    return "\n".join(summary_parts)


def collect_image_urls(articles: List[Dict[str, Any]]) -> List[str]:
    """
    收集所有图片URLs
    """
    image_urls = []

    for article in articles:
        # 文章图片
        article_images = article.get("article_images", [])
        image_urls.extend(article_images)

        # 评论图片
        comments = article.get("comments", [])
        for comment_group in comments:
            content_list = comment_group.get("content", [])
            for content_item in content_list:
                comment_images = content_item.get("images", [])
                image_urls.extend(comment_images)

    # 去重并过滤有效URL
    unique_urls = []
    seen = set()
    for url in image_urls:
        if url and url.startswith("http") and url not in seen:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


def collect_video_urls(articles: List[Dict[str, Any]]) -> List[str]:
    """
    收集所有视频URLs
    """
    video_urls = []

    for article in articles:
        # 文章视频
        article_videos = article.get("article_videos", [])
        video_urls.extend(article_videos)

        # 评论视频
        comments = article.get("comments", [])
        for comment_group in comments:
            content_list = comment_group.get("content", [])
            for content_item in content_list:
                comment_videos = content_item.get("videos", [])
                video_urls.extend(comment_videos)

    # 去重并过滤有效URL
    unique_urls = []
    seen = set()
    for url in video_urls:
        if url and url.startswith("http") and url not in seen:
            unique_urls.append(url)
            seen.add(url)

    return unique_urls


def generate_text_only_profile(user_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    生成纯文本用户画像（便捷函数）

    Args:
        user_data: 单个用户的历史数据
        api_key: API密钥

    Returns:
        用户画像JSON格式数据
    """
    return generate_user_profile(user_data, api_key, model_name="qwen-max", use_multimodal=False)


def generate_multimodal_profile(user_data: Dict[str, Any], api_key: str = None) -> Dict[str, Any]:
    """
    生成多模态用户画像（便捷函数）

    Args:
        user_data: 单个用户的历史数据
        api_key: API密钥

    Returns:
        用户画像JSON格式数据
    """
    return generate_user_profile(user_data, api_key, model_name="qwen-vl-max", use_multimodal=True)


if __name__ == "__main__":
    print("用户画像生成工具")
    print("=" * 50)

    # 选择分析模式
    print("请选择分析模式:")
    print("1. 纯文本分析 (快速, 只分析文字内容)")
    print("2. 多模态分析 (包含图片和视频内容分析)")
    print("3. 测试单个用户")
    print("4. 手动分析测试")

    choice = input("请输入选择 (1-4): ").strip()

    file_path_ = "data/weibo/user_histories.json"

    if choice == "1":
        # 纯文本批量处理
        print("\n开始纯文本批量处理...")
        process_user_histories(
            file_path=file_path_,
            output_path=file_path_.replace("user_histories.json", "user_profiles_text_only.json"),
            api_key=os.getenv("QWEN_API_KEY"),
            use_multimodal=False
        )

    elif choice == "2":
        # 多模态批量处理
        print("\n开始多模态批量处理...")
        process_user_histories(
            file_path=file_path_,
            output_path=file_path_.replace("user_histories.json", "user_profiles_multimodal.json"),
            model_name="qwen-vl-max",
            api_key=os.getenv("QWEN_API_KEY"),
            use_multimodal=True
        )

    elif choice == "3":
        # 测试单个用户
        test_file = "data/weibo/user_histories.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            user_histories = json.load(f)

        if user_histories:
            test_user = user_histories[0]  # 使用第一个用户
            print(f"\n测试用户: {test_user.get('uid')}")

            # 选择测试模式
            mode_choice = input("选择测试模式 - 1:纯文本 2:多模态: ").strip()
            use_multimodal = mode_choice == "2"

            print(f"使用{'多模态' if use_multimodal else '纯文本'}模式生成画像...")
            profile = generate_user_profile(
                test_user,
                api_key=os.getenv("QWEN_API_KEY"),
                use_multimodal=use_multimodal
            )
            print("\n生成的用户画像:")
            print(json.dumps(profile, ensure_ascii=False, indent=2))

    elif choice == "4":
        # 手动分析测试
        test_file = "data/weibo/user_histories.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            user_histories = json.load(f)

        if user_histories:
            test_user = user_histories[0]
            print(f"\n手动分析用户: {test_user.get('uid')}")

            manual_analysis = analyze_user_content_manually(test_user)
            print(json.dumps(manual_analysis, ensure_ascii=False, indent=2))

    else:
        print("无效选择，程序退出。")
