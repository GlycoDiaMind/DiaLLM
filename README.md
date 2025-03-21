展示demo:
首先完成预训练模型路径检查，再确认提示词之后启动，默认句子长度2048
注意内存释放

简要训练流程：
source /root/DiabetesPDiagLLM/.diabetesPDiagLLMVenv/bin/activate
cd DiabetesPDiagLLM/
python src/train/train.py

先在utils.py做数据预处理，再做训练