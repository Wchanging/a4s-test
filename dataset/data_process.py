import pandas as pd
import json


def load_data(file_path):
    """
    Load data from a CSV file and return a DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path, encoding='utf-8-sig', dtype=str)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def count_users_frequency(data, user_id_column='uid'):
    """
    Count the frequency of each user in the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing user data.

    Returns:
        pd.Series: A Series with user IDs as index and their frequencies as values.
    """
    if user_id_column not in data.columns:
        raise ValueError(f"DataFrame must contain {user_id_column} column.")

    return data[user_id_column].value_counts()


def select_user_data_json(data, user_id, user_id_column='uid'):
    """
    Select data for a specific user based on user ID.

    Args:
        data (pd.DataFrame): The DataFrame containing user data.
        user_id (str): The user ID to filter by.
        user_id_column (str): The column name for user IDs.

    Returns:
        pd.DataFrame: A DataFrame containing only the data for the specified user.
    """
    if user_id_column not in data.columns:
        raise ValueError(f"DataFrame must contain {user_id_column} column.")

    user_data = data[data[user_id_column] == user_id]
    if user_data.empty:
        print(f"No data found for user ID: {user_id}")
        return pd.DataFrame()
    # return a json format
    return user_data.to_json(orient='records', force_ascii=False, indent=2)


def save_user_data_csv(data, output_file_path, user_num, user_id_column='uid'):
    """
    Save data for a specific number of most frequent users to a CSV file.

    Args:
        data (pd.DataFrame): The DataFrame containing user data.
        output_file_path (str): The path to save the filtered data.
        user_num (int): The number of users to filter.
        user_id_column (str): The column name for user IDs.
    """
    if user_id_column not in data.columns:
        raise ValueError(f"DataFrame must contain {user_id_column} column.")

    # Get the most frequent users
    users_frequency = count_users_frequency(data, user_id_column)
    top_users = users_frequency.head(user_num).index

    # Filter the data for the top users
    filtered_data = data[data[user_id_column].isin(top_users)]

    # Save to CSV
    filtered_data.to_csv(output_file_path, index=False, encoding='utf-8-sig')


def save_user_data_json(data, output_file_path, user_num, user_id_column='uid'):
    """
    Save data for a specific number of most frequent users to a JSON file.

    Args:
        data (pd.DataFrame): The DataFrame containing user data.
        output_file_path (str): The path to save the filtered data.
        user_num (int): The number of users to filter.
        user_id_column (str): The column name for user IDs.
    """
    if user_id_column not in data.columns:
        raise ValueError(f"DataFrame must contain {user_id_column} column.")

    # Get the most frequent users
    users_frequency = count_users_frequency(data, user_id_column)
    top_users = users_frequency.head(user_num).index

    # Filter the data for the top users
    filtered_data = data[data[user_id_column].isin(top_users)]

    # Save to JSON
    filtered_data.to_json(output_file_path, orient='records', force_ascii=False, indent=2)


def get_user_history_data_json(meta_data, comments_data, uid, user_id_column='uid', article_id_column='article_id',
                               comment_id_column='comment_id', parent_comment_id_column='parent_comment_id',
                               content_column='content', img_column='img_urls', video_column='video_urls',
                               created_time='created_time', question_id_column='question_id', answer_id_column='answer_id'):
    """ Save user history data to a JSON file.
    Args:
        meta_data (pd.DataFrame): DataFrame containing article metadata.
        comments_data (pd.DataFrame): DataFrame containing comments data.
        uid (str): User ID to filter data.
        user_id_column (str): Column name for user IDs.
        article_id_column (str): Column name for article IDs.
        comment_id_column (str): Column name for comment IDs.
        parent_comment_id_column (str): Column name for parent comment IDs.
        content_column (str): Column name for comment content.
        img_column (str): Column name for image URLs.
        video_column (str): Column name for video URLs.
        created_time (str): Column name for creation time.
        question_id_column (str): Column name for question IDs (for Q&A content).
        answer_id_column (str): Column name for answer IDs (for Q&A content).

    Process:
        1. Load metadata and comments data from CSV files.
        2. Filter comments data for the specified user ID.
        3. Find out the article IDs associated with the user's comments.
        4. Create a mapping of article IDs to their comments, images, and videos. (Each article can have multiple comments.)
        5. Save the mapping to a JSON file.

    Example:
        {
            "uid": "xxxxxx",        
            "articles": [
                {
                "article_id": "xxxxxxx",
                "article_content": "Article content",
                "article_images": ["image_url_1", "image_url_2"],
                "article_videos": ["video_url_1"],
                "comments": [
                    {
                        "parent_comment": "",
                        "content": [
                            {
                                "content": "Top-level comment content",
                                "created_time": "2023-01-01 12:00:00",
                                "images": ["image_url_1"],
                                "videos": ["video_url_1"]
                            }
                        ]
                    },
                    {   
                        "parent_comment": "Parent comment content",
                        "content": [
                            {
                                "content": "First reply content",
                                "created_time": "2023-01-01 12:05:00",
                                "images": [],
                                "videos": []
                            },
                            {
                                "content": "Second reply content",
                                "created_time": "2023-01-01 12:10:00",
                                "images": ["reply_image.jpg"],
                                "videos": []
                            }
                        ]
                    }
                ]
                },
                {...}
            ]
        }
    """

    # Step 1: Load metadata and comments data from CSV files
    meta_data = meta_data if isinstance(meta_data, pd.DataFrame) else load_data(meta_data)
    comments_data = comments_data if isinstance(comments_data, pd.DataFrame) else load_data(comments_data)

    if meta_data is None or comments_data is None:
        print("Error: Failed to load data files")
        return

    # Step 2: Filter comments data for the specified user ID
    user_comments = comments_data[comments_data[user_id_column] == uid].copy()

    if user_comments.empty:
        print(f"No comments found for user ID: {uid}")
        return

    # Step 3: Find out the article IDs or question/answer IDs associated with the user's comments
    # Check if this is article-based or Q&A-based content
    has_articles = article_id_column in user_comments.columns and user_comments[article_id_column].notna().any()
    has_qa = (question_id_column in user_comments.columns and answer_id_column in user_comments.columns and
              user_comments[question_id_column].notna().any() and user_comments[answer_id_column].notna().any())

    if has_articles:
        content_ids = user_comments[article_id_column].unique()
        content_type = 'article'
        id_column = article_id_column
    elif has_qa:
        # For Q&A, combine question_id and answer_id as the content identifier
        user_comments['qa_id'] = user_comments[question_id_column].astype(str) + '_' + user_comments[answer_id_column].astype(str)
        content_ids = user_comments['qa_id'].unique()
        content_type = 'qa'
        id_column = 'qa_id'
    else:
        print(f"No valid content identifiers found for user {uid}")
        return

    # Step 4: Create a mapping of content IDs to their comments, images, and videos
    result = {
        "uid": uid,
        "articles": []
    }

    for content_id in content_ids:
        # Get content metadata based on content type
        if content_type == 'article':
            content_meta = meta_data[meta_data[article_id_column] == content_id]
            content_info = {
                "article_id": content_id,
                "content_type": "article",
                "article_content": "",
                "article_images": [],
                "article_videos": [],
                "comments": []
            }
        else:  # Q&A type
            question_id, answer_id = content_id.split('_')
            # First try to find by both question_id and answer_id
            content_meta = meta_data[
                (meta_data[question_id_column] == question_id) &
                (meta_data[answer_id_column] == answer_id)
            ]
            content_info = {
                "question_id": question_id,
                "answer_id": answer_id,
                "content_type": "qa",
                "question_content": "",
                "answer_content": "",
                "content_images": [],
                "content_videos": [],
                "comments": []
            }

        # Add content and media if available
        if not content_meta.empty:
            content_row = content_meta.iloc[0]
            if content_type == 'article':
                if content_column in content_meta.columns:
                    article_content = content_row.get(content_column, "")
                    # Handle NaN article content
                    if pd.isna(article_content):
                        article_content = ""
                    content_info["article_content"] = str(article_content)
                if img_column in content_meta.columns and pd.notna(content_row.get(img_column)):
                    img_urls = str(content_row[img_column]).strip('[]').split(',')
                    content_info["article_images"] = [url.strip().strip("'\"") for url in img_urls if url.strip()]
                if video_column in content_meta.columns and pd.notna(content_row.get(video_column)):
                    video_urls = str(content_row[video_column]).strip('[]').split(',')
                    content_info["article_videos"] = [url.strip().strip("'\"") for url in video_urls if url.strip()]
            else:  # Q&A type
                # Handle question content
                if 'title' in content_meta.columns:
                    question_content = content_row.get('title', "")
                    if pd.isna(question_content):
                        question_content = ""
                    content_info["question_content"] = str(question_content)

                # Handle answer content
                if content_column in content_meta.columns:
                    answer_content = content_row.get(content_column, "")
                    if pd.isna(answer_content):
                        answer_content = ""
                    content_info["answer_content"] = str(answer_content)

                # Handle media
                if img_column in content_meta.columns and pd.notna(content_row.get(img_column)):
                    img_urls = str(content_row[img_column]).strip('[]').split(',')
                    content_info["content_images"] = [url.strip().strip("'\"") for url in img_urls if url.strip()]
                if video_column in content_meta.columns and pd.notna(content_row.get(video_column)):
                    video_urls = str(content_row[video_column]).strip('[]').split(',')
                    content_info["content_videos"] = [url.strip().strip("'\"") for url in video_urls if url.strip()]

        # Get all comments for this content by the user
        if content_type == 'article':
            content_comments = user_comments[user_comments[article_id_column] == content_id]
        else:  # Q&A type
            content_comments = user_comments[user_comments['qa_id'] == content_id]

        # Group comments by parent_comment_id to consolidate replies
        comment_groups = {}

        for _, comment_row in content_comments.iterrows():
            comment_id = comment_row[comment_id_column]

            # Determine the grouping key (parent_comment_id or 'root' for top-level comments)
            parent_id = None
            if (parent_comment_id_column in comment_row and
                pd.notna(comment_row[parent_comment_id_column]) and
                    str(comment_row[parent_comment_id_column]).strip()):
                parent_id = str(comment_row[parent_comment_id_column]).strip()

            group_key = parent_id if parent_id else 'root'

            # Initialize group if not exists
            if group_key not in comment_groups:
                comment_groups[group_key] = {
                    'parent_comment_content': None,
                    'replies': []
                }

                # Get parent comment content if exists
                if parent_id:
                    parent_comment = comments_data[comments_data[comment_id_column] == parent_id]
                    if not parent_comment.empty:
                        parent_content = parent_comment.iloc[0].get(content_column, "")
                        # Handle NaN parent content
                        if pd.isna(parent_content):
                            parent_content = ""
                        comment_groups[group_key]['parent_comment_content'] = str(parent_content)

            # Process comment images and videos
            comment_images = []
            comment_videos = []

            if img_column in comment_row and pd.notna(comment_row[img_column]):
                img_urls = str(comment_row[img_column]).strip('[]').split(',')
                comment_images = [url.strip().strip("'\"") for url in img_urls if url.strip()]

            if video_column in comment_row and pd.notna(comment_row[video_column]):
                video_urls = str(comment_row[video_column]).strip('[]').split(',')
                comment_videos = [url.strip().strip("'\"") for url in video_urls if url.strip()]

            # Format creation time
            comment_time = None
            if created_time in comment_row and pd.notna(comment_row[created_time]):
                comment_time = str(comment_row[created_time])
                # transform the time format if needed
                # from timestamp to datetime string
                comment_time = pd.to_datetime(int(comment_time), unit='s').strftime('%Y-%m-%d %H:%M:%S')

            # Add reply to the group
            reply_info = {
                "comment_id": comment_id,
                "content": comment_row.get(content_column, ""),
                "created_time": comment_time,
                "images": comment_images,
                "videos": comment_videos
            }
            # Handle NaN content - check multiple possible causes
            if reply_info["content"] is None or pd.isna(reply_info["content"]):
                reply_info["content"] = ""
                print(f"Warning: Empty content found for comment_id {comment_id}, setting to empty string")

            # Convert to string to handle any remaining edge cases
            reply_info["content"] = str(reply_info["content"]) if reply_info["content"] is not None else ""

            comment_groups[group_key]['replies'].append(reply_info)

        # Convert grouped comments to the final unified structure
        for group_key, group_data in comment_groups.items():
            # Sort replies by creation time
            sorted_replies = sorted(group_data['replies'], key=lambda x: x.get('created_time', ''))

            # Create unified comment structure
            comment_group = {
                "parent_comment": group_data['parent_comment_content'] if group_key != 'root' else "",
                "content": []
            }

            # Add all replies to the content array
            for reply in sorted_replies:
                comment_group["content"].append({
                    "content": reply["content"],
                    "created_time": reply["created_time"],
                    "images": reply["images"],
                    "videos": reply["videos"]
                })

            content_info["comments"].append(comment_group)

        result["articles"].append(content_info)

    # Step 5: Sort comment groups by the earliest comment time in each group
    for article in result["articles"]:
        article["comments"] = sorted(article["comments"],
                                     key=lambda x: x["content"][0].get("created_time", "") if x["content"] else "")

    # Step 6: Return the result as a JSON string
    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return None


def save_users_history(meta_data, comments_data, output_file_path, user_num=10, user_id_column='uid',
                       article_id_column='article_id', question_id_column='question_id', answer_id_column='answer_id',
                       top_or_random='random'):
    """
    Save data for a specific number of most frequent users to a JSON file.

    Args:
        meta_data (pd.DataFrame): DataFrame containing article metadata.
        comments_data (pd.DataFrame): DataFrame containing comments data.
        output_file_path (str): The path to save the filtered data.
        user_num (int): The number of users to filter.
    """
    if not isinstance(meta_data, pd.DataFrame) or not isinstance(comments_data, pd.DataFrame):
        raise ValueError("Both meta_data and comments_data must be pandas DataFrames.")

    # Get the most frequent users
    users_frequency = count_users_frequency(comments_data, user_id_column=user_id_column)
    # top_users = users_frequency.head(user_num).index
    if top_or_random == 'top':
        top_users = users_frequency.head(user_num).index
    elif top_or_random == 'random':
        top_users = users_frequency.sample(n=user_num, random_state=42).index
    else:
        raise ValueError("top_or_random must be either 'top' or 'random'.")

    # Save each user's history data to JSON
    user_histories = []
    for uid in top_users:
        user_history_json = get_user_history_data_json(
            meta_data=meta_data,
            comments_data=comments_data,
            uid=uid,
            user_id_column=user_id_column,
            article_id_column=article_id_column,
            question_id_column=question_id_column,
            answer_id_column=answer_id_column,
        )
        if user_history_json:
            user_histories.append(json.loads(user_history_json))

    # Save the entire user history data to a JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(user_histories, f, ensure_ascii=False, indent=2)

    print(f"User histories saved to: {output_file_path}")


if __name__ == "__main__":
    # Example usage
    data_fold = "data/weibo/"
    meta_data = load_data(data_fold + 'contents.csv')
    comments_data = load_data(data_fold + 'comments.csv')

    save_users_history(
        meta_data=meta_data,
        comments_data=comments_data,
        output_file_path=data_fold + 'user_histories.json',
        user_num=100,
        user_id_column='uid',
        article_id_column='article_id',  # 文章类型
        question_id_column='question_id',  # 问答类型
        answer_id_column='answer_id'  # 问答类型
    )
