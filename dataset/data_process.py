import pandas as pd


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


def select_user_data(data, user_id, user_id_column='uid'):
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

    return data[data[user_id_column] == user_id]


def save_user_data(data, output_file_path, user_num, user_id_column='uid'):
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


if __name__ == "__main__":
    # Example usage
    file_path = 'data/weibo/comments.csv'
    data = load_data(file_path)

    if data is not None:
        # Count user frequencies
        user_frequencies = count_users_frequency(data)

        # Print the top 10 users by frequency
        print("Top 10 users by frequency:")
        print(user_frequencies.head(10))

    # if data is not None:
    #     user_id_column = 'uid'
    #     user_num = 100
    #     output_file_path = 'data/weibo/top_users_comments.csv'
    #     save_user_data(data, output_file_path, user_num, user_id_column)
