import pandas as pd
from datetime import datetime
import re
from itertools import combinations
from collections import defaultdict
from collections import Counter


def latilongi():
    # 读取 CSV 文件
    df = pd.read_csv('datasets/dataset_TSMC2014_NYC.csv')

    # 创建一个空的字典来存储每个 venueId 的第一个经纬度
    first_lat_lon = {}

    # 遍历 DataFrame
    for index, row in df.iterrows():
        venue_id = row['venueId']
        # 如果这是该 venueId 的第一个记录，存储它的经纬度
        if venue_id not in first_lat_lon:
            first_lat_lon[venue_id] = (row['latitude'], row['longitude'])
        else:
            # 如果这个 venueId 已经有了经纬度，替换当前行的经纬度
            df.at[index, 'latitude'], df.at[index, 'longitude'] = first_lat_lon[venue_id]

    df.to_csv('mod_2014_NYC.csv', index=False)

def venuegroup(df):
    # 根据 venueId 分组，并计算每个地点的访问人数
    grouped_df = df.groupby('venueId').agg(
        venueCategoryId=('venueCategoryId', 'first'),
        venueCategory=('venueCategory', 'first'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first'),
        count=('userId', 'count')
    ).reset_index()

    grouped_df.to_csv('overallcount_2014_NYC.csv', index=False)

def usergroup(userId):
    user_df = df[df['userId'] == userId]
    def parsed(timestamp):
        # 正则表达式匹配
        match = re.match(r'(\w{3}) (\w{3}) (\d{2}) (\d{2}:\d{2}:\d{2}) \+0000 (\d{4})', timestamp)
        if match:
            day_of_week, month, day, time, year = match.groups()
            # 解析月份
            month_number = datetime.strptime(month, '%b').month
            # 组合成标准格式
            standard_timestamp = f'{year}-{month_number:02d}-{day} {time}'
            return standard_timestamp
        else:
            return None
        
    user_df.loc[:, 'utcTimestamp'] = user_df['utcTimestamp'].apply(
        parsed)
    user_df.to_csv(f'code/Visualization/{userId}_2014_NYC.csv', index=False)

def findoverlap(df, community):
    # 筛选出Community字段等于3的Node
    nodes = community[community['Community'] == 18]['Node']

    # 在df中筛选出符合条件的userId
    filtered_df = df[df['userId'].isin(nodes)]

    # 构建每个userId对应的venueId集合
    user_venues = defaultdict(set)
    for _, row in filtered_df.iterrows():
        user_venues[row['userId']].add(row['venueId'])

    # 计算所有userId对的venueId重叠次数
    overlap_counts = Counter()
    for (user1, venues1), (user2, venues2) in combinations(user_venues.items(), 2):
        overlap = len(venues1 & venues2)
        if overlap > 0:
            overlap_counts[(user1, user2)] = overlap

    # 找到重叠次数最多的userId对
    most_common_users = overlap_counts.most_common(1)

    return most_common_users[0][0] if most_common_users else []



def find2(df, community, comid):
    nodes = community[community['Community'] == comid]['Node']
    df = df[df['userId'].isin(nodes)]
    venuegroup(df)



df = pd.read_csv('datasets/dataset_TSMC2014_NYC.csv')
community = pd.read_csv('community_partition_2.csv')


# print(findoverlap(df, community))