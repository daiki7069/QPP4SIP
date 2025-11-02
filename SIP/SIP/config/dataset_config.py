"""
INSCITデータセット設定
"""
from sklearn.preprocessing import MultiLabelBinarizer


class Config():
    """INSCITデータセット設定を管理するクラス"""
    
    def __init__(self, args):
        # INSCITデータセットの設定
        self.max_target_length = 1  # SIP予測は単一ラベル
        self.pad_index = 0
        self.soa_index = 1
        self.eoa_index = 2
        
        # INSCITのresponseTypeラベル
        self.response_types = [
            'directAnswer',
            'clarification', 
            'noAnswerButRelevantInfo',
            'noAnswerNoRelevantInfo'
        ]
        
        # responseTypeからIDへのマッピング
        self.response_type2id = {
            'directAnswer': 0,
            'clarification': 1,
            'noAnswerButRelevantInfo': 2,
            'noAnswerNoRelevantInfo': 3
        }
        
        # IDからresponseTypeへのマッピング
        self.id2response_type = {
            0: 'directAnswer',
            1: 'clarification',
            2: 'noAnswerButRelevantInfo',
            3: 'noAnswerNoRelevantInfo'
        }
        
        # SIPラベルの定義
        # clarification -> SIP: 1 (システムがイニシアチブを取る)
        # その他 -> SIP: 0 (システムがイニシアチブを取らない)
        self.sip_labels = {
            'directAnswer': 0,
            'clarification': 1,
            'noAnswerButRelevantInfo': 0,
            'noAnswerNoRelevantInfo': 0
        }
        
        # MultiLabelBinarizerの設定
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.response_types])
        
        # データセット固有のパラメータ
        self.dataset_name = "INSCIT"
        self.num_classes = len(self.response_types)