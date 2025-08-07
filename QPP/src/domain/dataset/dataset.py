from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class SeedArticle:
    """記事情報"""
    title: str
    topic: str
    level: int

@dataclass
class RetrievedEvidence:
    """検索された証拠情報"""
    passage_id: str
    passage_text: str
    passage_titles: List[str]
    score: float

@dataclass
class DialogueTurn:
    """会話ターン"""
    query: str
    resolved_query: str
    retrieved_evidence: List[RetrievedEvidence]

@dataclass
class DialogueData:
    """個別のダイアログデータ"""
    seed_article: SeedArticle
    turns: List[DialogueTurn]

@dataclass
class InscitRetrievedDataset:
    """Inscit Retrieved Dataset エンティティ"""
    dialogues: Dict[str, DialogueData]
    
    @classmethod
    def from_json_file(cls, file_path: str) -> 'InscitRetrievedDataset':
        """JSONファイルからInscitRetrievedDatasetを作成"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dialogues = {}
        for dialogue_id, dialogue_data in data.items():
            # SeedArticleの作成
            seed_article = SeedArticle(
                title=dialogue_data['seedArticle']['title'],
                topic=dialogue_data['seedArticle']['topic'],
                level=dialogue_data['seedArticle']['level']
            )
            
            # Turnsの作成
            turns = []
            for turn_data in dialogue_data['turns']:
                retrieved_evidence = []
                for evidence_data in turn_data['retrieved_evidence']:
                    evidence = RetrievedEvidence(
                        passage_id=evidence_data['passage_id'],
                        passage_text=evidence_data['passage_text'],
                        passage_titles=evidence_data['passage_titles'],
                        score=evidence_data['score']
                    )
                    retrieved_evidence.append(evidence)
                
                turn = DialogueTurn(
                    query=turn_data['query'],
                    resolved_query=turn_data['resolvedQuery'],
                    retrieved_evidence=retrieved_evidence
                )
                turns.append(turn)
            
            dialogue = DialogueData(
                seed_article=seed_article,
                turns=turns
            )
            dialogues[dialogue_id] = dialogue
        
        return cls(dialogues=dialogues)
    
    def get_dialogue(self, dialogue_id: str) -> DialogueData:
        """指定したIDのダイアログを取得"""
        return self.dialogues[dialogue_id]
    
    def get_all_dialogue_ids(self) -> List[str]:
        """全てのダイアログIDを取得"""
        return list(self.dialogues.keys())
    
    def __len__(self) -> int:
        """ダイアログ数を返す"""
        return len(self.dialogues)
    