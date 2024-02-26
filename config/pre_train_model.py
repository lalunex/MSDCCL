pre_train_model_dict = {
    'amazon-sports-outdoors': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-14-44.pth',
        'NARM': 'saved/NARM-May-17-2022_13-16-07.pth',
        'Caser': 'saved/Caser-May-17-2022_13-17-42.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-20-26.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-27-07.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-20-57.pth',
        'DSAN': 'saved/DSAN-May-19-2022_23-25-56.pth'
    },
    'ml-100k': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-42.pth',
        'NARM': 'saved/NARM-May-19-2022_19-31-33.pth',
        'Caser': 'saved/Caser-May-17-2022_13-13-56.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-14-04.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-14-14.pth',
        'STAMP': 'saved/STAMP-May-17-2022_13-14-49.pth'
        # 将GRU4Rec训练好的embedding又拿去训练GRU4Rec，效果不变
    },
    'yelp': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-15-01.pth',
        'NARM': 'saved/NARM-May-17-2022_13-17-59.pth',
        'Caser': 'saved/Caser-May-17-2022_13-26-00.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-30-05.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-45-49.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-27-11.pth'
    },
    'amazon-beauty': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-26.pth',
        'NARM': 'saved/NARM-May-17-2022_13-14-20.pth',
        'Caser': 'saved/Caser-May-17-2022_13-16-13.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-17-49.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-21-48.pth',
        'STAMP': 'saved/STAMP-May-17-2022_14-05-50.pth'
    },
    'amazon-clothing-shoes-jewelry': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-32-16.pth',
        'NARM': 'saved/NARM-May-17-2022_13-34-29.pth',
        'Caser': 'saved/Caser-May-17-2022_13-37-12.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-40-44.pth',
        # 'BERT4Rec': 'saved/Caser-May-17-2022_13-37-12.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-46-02.pth',
        'STAMP': 'saved/STAMP-May-17-2022_15-03-23.pth'
        #使用 Caser的embedding尝试训练Bert4Rec
    },
    'ml-1m': {
        'GRU4Rec': 'saved/GRU4Rec-May-17-2022_13-13-41.pth',
        'NARM': 'saved/NARM-May-17-2022_13-14-36.pth',
        'Caser': 'saved/Caser-May-17-2022_13-15-33.pth',
        'BERT4Rec': 'saved/BERT4Rec-May-17-2022_13-16-38.pth',
        'SASRec': 'saved/SASRec-May-17-2022_13-27-12.pth',
        'STAMP': 'saved/STAMP-May-17-2022_13-39-43.pth'
        # 'STAMP': 'saved/BERT4Rec-May-17-2022_13-16-38.pth'  # 用bert4rec的embedding比用stamp自己学出来的性能要好
    }

}