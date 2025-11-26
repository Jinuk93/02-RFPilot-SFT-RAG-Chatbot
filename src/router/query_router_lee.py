# src/router/query_router.py

import logging

logger = logging.getLogger(__name__)

class QueryRouter:
    """Query를 RAG vs Direct로 라우팅"""

    def __init__(self):
        # 키워드 정의
        self.greeting_keywords = [
            "안녕", "hi", "hello", "반가워", "처음", "인사"
        ]

        self.thanks_keywords = [
            "고마워", "감사", "thanks", "고맙", "땡큐"
        ]

        self.document_keywords = [
            # 돈 관련
            "예산", "비용", "금액", "원", "만원", "억", "억원",
            # 일정 관련
            "기한", "마감", "언제", "기간", "납기", "일정",
            # 문서 관련
            "요구사항", "제출", "서류", "양식", "평가", "rfp",
            # 조직 관련
            "발주", "기관", "담당자", "연락처", "부처", "지자체",
            # 사업/계약 관련
            "사업", "사업명", "과업", "범위", "목적", "계약", "입찰",
            "공고", "프로젝트", "위탁", "용역", "협상", "제안"
        ]

    def classify(self, query: str) -> dict:
        query_lower = query.lower()
        query_length = len(query)

        # 짧은 질문일 때만 인사/감사 체크
        if query_length < 25:
            if any(kw in query_lower for kw in self.thanks_keywords):
                return {
                    'type': 'thanks',
                    'confidence': 0.9,
                    'reason': '감사 인사 감지'
                }

            if any(kw in query_lower for kw in self.greeting_keywords):
                return {
                    'type': 'greeting',
                    'confidence': 0.9,
                    'reason': '인사 감지'
                }

        # 문서 관련 판별 (키워드 또는 숫자+사업 맥락)
        if any(kw in query_lower for kw in self.document_keywords):
            match_count = sum(1 for kw in self.document_keywords if kw in query_lower)
            confidence = min(0.7 + 0.05 * match_count, 0.95)
            return {
                'type': 'document',
                'confidence': confidence,
                'reason': f'문서 키워드 {match_count}개 감지'
            }

        # 숫자와 행정 용어가 혼합된 경우 약한 문서 추정
        if any(ch.isdigit() for ch in query) and any(term in query_lower for term in ["사업", "과업", "계획"]):
            return {
                'type': 'document',
                'confidence': 0.65,
                'reason': '숫자와 사업 키워드 동시 감지'
            }

        # 기본값
        return {
            'type': 'out_of_scope',
            'confidence': 0.4,
            'reason': 'RFP 관련 키워드 미감지'
        }