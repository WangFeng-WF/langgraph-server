import os


class AppConfig:
    ALLOWED_BUSINESS_DOMAINS = ["����", "�ɹ�", "����", "��Ӧ��"]
    LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ca949c46e4904479927923a41562d4d3")
    LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL_NAME = "qwen-max-latest"

    STEP_MAP = {
        1: "���ʶ���", 2: "���㹫ʽ", 3: "����ά����ָ��", 4: "������Դ",
        5: "ָ�����-�����ѯSQL", 6: "��ϸ����-ʵʱ����SQL",
        7: "ָ���������", 8: "��ϸ��������"
    }
