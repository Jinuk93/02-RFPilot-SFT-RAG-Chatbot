# src/router/query_router.py

import logging

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Query를 RAG vs Direct로 라우팅 (하이브리드 버전)
    
    improved + lee 버전의 장점 결합:
    - improved: out_of_scope 키워드로 명확한 비RFP 질문 감지
    - lee: 숫자 + 사업 키워드 조합으로 맥락 파악
    """

    def __init__(self):
        # 인사 키워드
        self.greeting_keywords = [
            "안녕", "hi", "hello", "반가워", "처음", "인사"
        ]

        # 감사 키워드
        self.thanks_keywords = [
            "고마워", "감사", "thanks", "고맙", "땡큐"
        ]

        # RFP/입찰 관련 키워드
        self.document_keywords = [
            # 돈 관련
            "예산", "비용", "금액", "원", "만원", "억", "억원",
            # 일정 관련
            "기한", "마감", "언제", "기간", "납기", "일정",
            # 문서 관련
            "요구사항", "제출", "서류", "양식", "평가", "rfp", "제안서",
            # 조직 관련
            "발주", "기관", "담당자", "연락처", "부처", "지자체",
            # 사업/계약 관련
            "사업", "사업명", "과업", "범위", "목적", "계약", "입찰",
            "공고", "프로젝트", "위탁", "용역", "협상", "제안",
            # 제도/규정 관련
            "법", "규정", "기준", "조건", "중소기업", "대기업"
        ]
        
        # ✅ out_of_scope 키워드 (improved 버전에서 가져옴)
        self.out_of_scope_keywords = [
            # 음식
            "점심", "저녁", "아침", "식사", "밥", "메뉴", "맛집", "음식", "요리",
            # 날씨/일상
            "날씨", "기온", "비", "눈", "추워", "더워",
            # 엔터테인먼트
            "영화", "드라마", "게임", "노래", "음악", "유튜브",
            # 여행/취미
            "여행", "관광", "휴가", "취미", "운동", "등산",
            # 금융/투자 (RFP와 무관)
            "주식", "코인", "비트코인", "투자", "펀드", "부동산",
            # 기타
            "사랑", "연애", "데이트", "친구", "가족"
        ]

    def classify(self, query: str) -> dict:
        """
        쿼리 분류
        
        Returns:
            dict: {
                'type': 'greeting' | 'thanks' | 'document' | 'out_of_scope',
                'confidence': 0.0~1.0,
                'reason': str
            }
        """
        query_lower = query.lower()
        query_length = len(query)
        
        # ✅ 1. 명확한 out_of_scope 먼저 체크 (improved 로직)
        for keyword in self.out_of_scope_keywords:
            if keyword in query_lower:
                logger.info(f"🚫 out_of_scope 감지: '{keyword}' 키워드")
                return {
                    'type': 'out_of_scope',
                    'confidence': 0.95,
                    'reason': f'비RFP 키워드 감지: {keyword}'
                }

        # 2. 짧은 질문일 때만 인사/감사 체크 (lee의 25자 기준 사용)
        if query_length < 25:
            # 감사
            if any(kw in query_lower for kw in self.thanks_keywords):
                logger.info(f"🙏 thanks 감지")
                return {
                    'type': 'thanks',
                    'confidence': 0.90,
                    'reason': '감사 인사 감지'
                }
            
            # 인사
            if any(kw in query_lower for kw in self.greeting_keywords):
                logger.info(f"👋 greeting 감지")
                return {
                    'type': 'greeting',
                    'confidence': 0.90,
                    'reason': '인사 감지'
                }

        # 3. RFP/문서 관련 키워드 체크 (동적 신뢰도)
        document_matches = sum(1 for kw in self.document_keywords if kw in query_lower)
        
        if document_matches > 0:
            # 매칭된 키워드 수에 따라 신뢰도 조정
            confidence = min(0.70 + (document_matches * 0.05), 0.95)
            logger.info(f"📄 document 감지: {document_matches}개 키워드 매칭")
            return {
                'type': 'document',
                'confidence': confidence,
                'reason': f'RFP 키워드 {document_matches}개 감지'
            }

        # ✅ 4. 숫자 + 사업 키워드 조합 체크 (lee 로직)
        # "12개월 사업", "5억원 프로젝트" 같은 맥락 파악
        has_number = any(ch.isdigit() for ch in query)
        business_terms = ["사업", "과업", "계획", "프로젝트", "용역"]
        has_business = any(term in query_lower for term in business_terms)
        
        if has_number and has_business:
            logger.info(f"🔢 document 감지: 숫자 + 사업 키워드 조합")
            return {
                'type': 'document',
                'confidence': 0.65,
                'reason': '숫자와 사업 키워드 동시 감지'
            }

        # 5. 기본값: out_of_scope (improved의 0.6 사용)
        logger.info(f"🚫 out_of_scope (기본값): RFP 키워드 없음")
        return {
            'type': 'out_of_scope',
            'confidence': 0.60,
            'reason': 'RFP 관련 키워드 미감지'
        }