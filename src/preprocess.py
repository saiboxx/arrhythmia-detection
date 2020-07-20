import re
from typing import Tuple
import pandas as pd
from tqdm import tqdm
from src.utils import Timer, load_config, save_pickle


def main():
    with Timer('Loading config'):
        cfg = load_config()

    with Timer('Loading tweets'):
        tweets = load_raw_data(cfg['RAW_DATA_PATH'])

    with Timer('Cleaning sentences'):
        tweet_text = cleanse_sentences(list(tweets['text']))

    with Timer('Mapping characters to integers'):
        tweet_enc, map_char_to_int, map_int_to_char = map_tweets_to_int(tweet_text)

    with Timer('Producing dataset'):
        tweet_train, tweet_label = produce_dataset(tweet_enc)

    with Timer('Save dataset and mapping tables'):
        save_pickle(tweet_train, cfg['PROCESSED_DATA_DIR'] + '/train.pkl')
        save_pickle(tweet_label, cfg['PROCESSED_DATA_DIR'] + '/label.pkl')
        save_pickle(map_char_to_int, cfg['PROCESSED_DATA_DIR'] + '/map_char_to_int.pkl')
        save_pickle(map_int_to_char, cfg['PROCESSED_DATA_DIR'] + '/map_int_to_char.pkl')


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw json file from trumptwitterarchive.com containing tweets.
    :param filepath: Filepath pointing to a .json file with tweets
    :return: Dataframe with the respective tweets.
    """
    full_data = pd.read_json(filepath, orient='records')
    full_data['source'] = full_data['source'].astype('string')
    full_data['text'] = full_data['text'].astype('string')
    full_data['is_retweet'] = full_data['is_retweet'].astype('bool')
    return full_data


def cleanse_sentences(tweet_list: list) -> list:
    """
    Runs checks for the tweets so that most special characters,
    emojis, links and retweet tags are removed.
    :param tweet_list: List containing tweets
    :return: Cleansed list of strings.
    """
    result = []
    for tweet in tweet_list:
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'(RT|rt)( @\w*)?[: ]', '', tweet)
        tweet = tweet.lower()
        tweet = (tweet.encode('ascii', 'ignore')).decode("utf-8")
        tweet = tweet.replace("\\", "")
        tweet = tweet.replace("\n", "")
        tweet = tweet.replace("\r", "")
        tweet = tweet.replace("}", "")
        tweet = tweet.replace("{", "")
        tweet = tweet.replace("[", "")
        tweet = tweet.replace("]", "")
        tweet = tweet.replace("*", "")
        tweet = tweet.replace("_", "")
        tweet = tweet.replace("/", "")
        tweet = tweet.replace("`", "")
        tweet = tweet.replace("|", "")
        tweet = tweet.replace("~", "")
        tweet = tweet.replace("...", "")
        tweet = tweet.replace("....", "")
        tweet = tweet.strip()

        if tweet != "":
            result.append(tweet)

    return result


def map_tweets_to_int(tweet_list: list) -> Tuple[list, dict, dict]:
    """
    Maps the cleansed tweets to lists of integer, which is better for further
    processing. The mapping table is also returned.
    :param text_list: List containing tweets
    :return: Encoded tweets and mapping dictionaries.
    """
    unique_chars = sorted({char for word in tweet_list for char in word})
    mapping_char_to_int = {char: i for i, char in enumerate(unique_chars)}
    mapping_int_to_char = {i: char for i, char in enumerate(unique_chars)}

    encoded_tweets = []
    for tweet in tweet_list:
        tweet = list(tweet)
        tweet = [mapping_char_to_int[char] for char in list(tweet)]
        encoded_tweets.append(tweet)

    return encoded_tweets, mapping_char_to_int, mapping_int_to_char


def produce_dataset(tweet_list: list) -> Tuple[list, list]:
    """
    Generates the training dataset. The label is a letter and the features
    consist of a sequence of chars coming before it, e.g. trum -> p, preside -> n.
    :param tweet_list: List w/ tweets, best encoded as integers
    :return: Tuple containing the dataset and accompanying labels.
    """
    dataset = []
    label = []
    for tweet in tqdm(tweet_list):
        for i, char in enumerate(tweet):
            if i != 0:
                dataset.append(tweet[:i])
                label.append(char)

    print("Dataset consists of {} samples.".format(len(dataset)))
    return dataset, label


if __name__ == '__main__':
    main()
