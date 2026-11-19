"""RAG 기반 답변 생성. 검증 대상이 되는 답변을 만듭니다 (로컬 Ollama 사용).

지금은 `scripts/generate_samples.py` 가 ai-server 의 Ollama(`gemma3:4b`)로
검토용 답변을 만듭니다. 이 모듈은 그것을 패키지 안으로 옮길 자리입니다.

검증 쪽(`verifier`, `api`)은 생성기 없이도 돕니다. 답변 텍스트를 받으면
되기 때문입니다. 그래서 뒤로 미뤄 두었습니다.
"""
