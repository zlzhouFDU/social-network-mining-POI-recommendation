from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

train_df = pd.read_csv("D:/social_network_Fall_23/NYC_train.csv")
sim_df = pd.read_csv("D:/social_network_Fall_23/similar_users_2.csv")

# [1, 5, 8, 11]

data_idx = 11
# target_data = sim_df[sim_df['user_idx']==data_idx]
target_data = sim_df.iloc[data_idx]
target_user = target_data['user_idx'].item()
target_sim_1 = target_data['similar_user_1'].item()
target_sim_2 = target_data['similar_user_2'].item()
target_sim_3 = target_data['similar_user_3'].item()

target_user_cat = list(train_df[train_df['user_id'] == target_user]['POI_catname'])
text_data = target_user_cat
text_data_str_user = ' '.join(text_data)
print(text_data_str_user)

target_sim_cat_1 = list(train_df[train_df['user_id'] == target_sim_1]['POI_catname'])
text_data = target_sim_cat_1
text_data_str_sim_1 = ' '.join(text_data)
print(text_data_str_sim_1)

target_sim_cat_2 = list(train_df[train_df['user_id'] == target_sim_2]['POI_catname'])
text_data = target_sim_cat_2
text_data_str_sim_2 = ' '.join(text_data)
print(text_data_str_sim_2)

target_sim_cat_3 = list(train_df[train_df['user_id'] == target_sim_3]['POI_catname'])
text_data = target_sim_cat_3
text_data_str_sim_3 = ' '.join(text_data)
print(text_data_str_sim_3)

# # 输入文本数据，可以是字符串或列表
# text_data_user = "Café Office Office Café Café Café Office College Academic Building Café Office Office Café Office Office Sandwich Place Beer Garden Food & Drink Shop Ferry Other Great Outdoors Café Office Café Office Café Office Café Office Office Movie Theater Café Office Café Office Café Office"
# text_data_sim_1 = "Bar Bar Gym / Fitness Center Performing Arts Venue Italian Restaurant Office Gym / Fitness Center Gym / Fitness Center Movie Theater Gym / Fitness Center"
# text_data_sim_2 = "Event Space Event Space Park Event Space Food & Drink Shop Coffee Shop Event Space Taco Place Park Event Space Food & Drink Shop Music Venue Café Event Space Park Event Space Event Space Park Music Venue Event Space Park Food Truck Movie Theater Food Truck Event Space Event Space Performing Arts Venue Park Event Space Event Space Food Truck Movie Theater Event Space Event Space Event Space Food Truck Park Event Space Event Space Neighborhood Park Event Space BBQ Joint Event Space Park Event Space Park Café Event Space"
# text_data_sim_3 = "Electronics Store Art Museum Garden Electronics Store Ice Cream Shop Fried Chicken Joint"

# 创建WordCloud对象
wordcloud_user = WordCloud(width=800, height=400, background_color='white').generate(text_data_str_user)
wordcloud_sim_1 = WordCloud(width=800, height=400, background_color='white').generate(text_data_str_sim_1)
wordcloud_sim_2 = WordCloud(width=800, height=400, background_color='white').generate(text_data_str_sim_2)
wordcloud_sim_3 = WordCloud(width=800, height=400, background_color='white').generate(text_data_str_sim_3)

# 可以选择设置其他参数，例如字体、最大词数等
# wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, font_path='path/to/font.ttf').generate(text_data)

# # 绘制词云图
# plt.figure(figsize=(10, 8))
# plt.imshow(wordcloud_user, interpolation='bilinear')
# plt.axis('off')  # 不显示坐标轴
# plt.show()

# 第一个子图
plt.subplot(2, 2, 1)
plt.imshow(wordcloud_user, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title("Target User")

# 第二个子图
plt.subplot(2, 2, 2)
plt.imshow(wordcloud_sim_1, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title("1-st Similar User")

# 第三个子图
plt.subplot(2, 2, 3)
plt.imshow(wordcloud_sim_2, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title("2-nd Similar User")

# 第四个子图
plt.subplot(2, 2, 4)
plt.imshow(wordcloud_sim_3, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.title("3-rd Similar User")

# 调整布局
plt.tight_layout()

# 显示图形

plt.savefig("world_cloud_4.png")

