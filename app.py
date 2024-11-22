from flask import Flask, request, jsonify, render_template
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

app = Flask(__name__)

model_dir = "lsylsy99/m2m_kovi_translate"
# model_dir = "mytrans_model"
tokenizer = M2M100Tokenizer.from_pretrained(model_dir)
model = M2M100ForConditionalGeneration.from_pretrained(model_dir)

@app.route("/create", methods=["POST"])
def create():
    json_input = request.get_json()
    type = json_input.get("selectedType", "")
    
    if type == "":
        return jsonify({"error": "상황 입력 안됨"}), 400  # 상황 미입력 에러
    
    data = json_input.get("data", None)
    if data is None:
        return jsonify({"error": "data 입력 안됨"}), 400  # 데이터 미입력 에러

    # 처리 타입 매핑
    type_mapping = {
        'Khoản vay': ['amount', 'date', 'partialReturn', 'partialAmount', 'partialDate'],
        'Hoàn trả tiền đặt cọc thuê nhà': ['depositAmount', 'endReason', 'contractDate', 'returnDate'],
        'Hủy hợp đồng thuê' : ['depositAmount', 'endReason', 'contractDate', 'returnDate'],
        'Yêu cầu thực hiện hợp đồng' : ['contractDate', 'contractName', 'signContent', 'obligation']  
    }
    
    if type not in type_mapping:
        return jsonify({"error": f"지원되지 않는 type: {type}"}), 400  # 지원하지 않는 type 에러
    
    information = {key: data.get(key) for key in type_mapping[type]}   
    
    ##내용증명 생성
    def generate_naeyong(type, information, json_input=None):
        base_text = [
            "1. 귀하(수신인, 이하 '귀하'라고 한다)의 무궁한 발전을 기원합니다.",]
        
        if type == 'Khoản vay':  # 대여금
            title = '대여금 변제 청구'
            if bool(information['partialReturn']):
                return title, base_text[:1] + [
                    f"2. 본 발신인은 귀하에게 아래와 같이 {information['amount']}원을 빌려주었습니다.",
                    f"3. 그러나 귀하는 아래와 같이 일부 변제만 하고 나머지 원금 {int(information['amount']) - int(information['partialAmount'])} 원을 변제하지 않고 있습니다.",
                    f"가. {information['partialDate']}일에 {information['partialAmount']}원을 변제 받음",
                    "4. 따라서 본 발신인은 귀하에게 즉시, 본 발신인에게 위 미지급 대여금을 지급 변제할 것을 촉구합니다.",
                    "그렇지 아니하면 본 발신인은 귀하의 재산에 대하여 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 민사소송법 등에 따른 민사 소송은 물론 형법이나 관련 특별법 등에 따른 형사상 고소 및 행정제재 등의 법적 조치를 할 것을 엄중히 경고합니다.",
                    "이 경우 귀하는 대여금 변제 의무에 따른 원금, 지연손해금은 물론, 소송비용까지 귀하가 부담하게 될 것입니다.",
                    "게다가 본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 관한 규정에 따라 연 12%가 될 것입니다."
                ]
            else:
                return title, base_text[:1] + [
                    f"2. 본 발신인은 귀하에게 아래와 같이 {information['amount']}원을 빌려주었습니다.",
                    f"3. 그러나 귀하는 변제 기일이 지난 현재까지도 {information['amount']}원을 본 발신인의 수차례에 걸친 변제의 독촉에도 불구하고 변제하지 않고 있습니다.",
                    "4. 따라서 본 발신인은 귀하에게 즉시, 본 발신인에게 위 미지급 대여금을 지급 변제할 것을 촉구합니다.",
                    "그렇지 아니하면 본 발신인은 귀하의 재산에 대하여 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 민사소송법 등에 따른 민사 소송은 물론 형법이나 관련 특별법 등에 따른 형사상 고소 및 행정제재 등의 법적 조치를 할 것을 엄중히 경고합니다.",
                    "이 경우 귀하는 대여금 변제 의무에 따른 원금, 지연손해금은 물론, 소송비용까지 귀하가 부담하게 될 것입니다.",
                    "게다가 본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 관한 규정에 따라 연 12%가 될 것입니다."
                ]

        elif type == 'Hoàn trả tiền đặt cọc thuê nhà':  # 임대차 보증금 반환 이건 내용증명 로직 잘못됐음.
            title = ""
            # additional_text = [
            #     f"2. 본 발신인은 귀하와 {information['contractDate']}일 임대차계약을 체결하였으며, 이에 귀하는 본 발신인에게 임대차 보증금 {information['depositAmount']}원을 반환할 의무가 있습니다.",
            #     f"3. 본 발신인은 귀하가 {information['returnDate']}일까지 보증금을 반환할 것을 촉구합니다. 이를 이행하지 않을 경우, 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 민사소송법 등에 따른 법적 조치를 할 것을 경고합니다.",
            #     "이 경우 귀하는 원금, 지연손해금, 소송비용까지 부담하게 될 것입니다.",
            #     "게다가 본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 따라 연 12%가 될 것입니다."
            # ]f
            return None

        elif type == 'Hủy hợp đồng thuê':  # 임대차 계약 해지
            title = '임대차 계약해지'
            sender_address = json_input.get("senderInfo")['address']
            if information['endReason'] == 'Kết thúc thời hạn':
                return title, base_text[:1] + [
                    f"2. 본 발신인은 귀하와 아래와 같이 임대차계약을 체결하였으며, 이에 귀하는 본 발신인에게 임대차 보증금 반환 의무가 있는바, 이에 대한 이행을 촉구합니다.",
                    f"가. 본 발신인은 귀하와 {information['contractDate']}일 {sender_address}에 관하여, 임대차 보증금 {information['depositAmount']}원으로 하는 임대차 계약을 체결하였습니다.",
                    f"나. 이에 본 발신인은 본 임대차계약을 갱신하지 않고자 하는바 귀하는 본 발신인에게 본 임대차 계약의 종료 시 임대차 보증금 1,000,000원을 반환할 의무가 있다고 할것입니다.",
                    f"3. 본 발신인은 귀하가 {information['returnDate']}일 까지 본 발신인에게 임대차 보증금을 반환할 것을 촉구하며 이를 이행하지 않을 때는 본 발신인은 귀하의 재산에 대하여 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 임차권등기명령, 민사소송법 등에 따른 보증금반환 청구 소송 등의 법적 조치를 할 것을 엄중히 경고합니다.",
                    "이 경우, 귀하는 임대차 보증금 반환 의무에 따른 원금, 이자 및 지연 손해금은 물론, 소송비용까지 귀하가 부담하게 될 것입니다.",
                    "게다가 본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 관한 규정에 따라 연 12%가 될 것입니다.",
                    "4. 본 발신인이 귀하에게 위와 같은 법적 조치를 취하기 전에 귀하는 기한 내에 귀하의 명백한 임대차 보증금 반환 의무를 이행하여 본 건을 원만하게 해결하시기를 마지막으로 말씀드립니다."
                ]
            elif information['endReason'] == 'Yêu cầu chấm dứt hợp đồng':
                return title, base_text[:1] + [
                    f"2. 본 발신인은 귀하와 아래와 같이 임대차계약을 체결하였으며, 이에 귀하는 본 발신인에게 임대차 보증금 반환 의무가 있는바, 이에 대한 이행을 촉구합니다.",
                    f"가. 본 발신인은 귀하와 {information['contractDate']}일 {sender_address}에 관하여, 임대차 보증금 {information['depositAmount']}원으로 하는 임대차 계약을 체결하였습니다.",
                    f"나. 이후 본 건 임대차계약은 묵시적으로 갱신되어 현재까지 임대차 계약이 계속되고 있습니다. 이 경우 임대차 보호와 관련한 법령 등에 따라 임차인은 임대차계약을 자유로이 해지할 수 있고, 귀하가 해지 통지를 받은 날로부터 3개월이 지나면 임대차계약은 종료된다고 할 것입니다.",
                    f"3. 본 발신인은 귀하가 {information['returnDate']}일 까지 본 발신인에게 임대차 보증금을 반환할 것을 촉구하며 이를 이행하지 않을 때는 본 발신인은 귀하의 재산에 대하여 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 임차권등기명령, 민사소송법 등에 따른 보증금반환 청구 소송 등의 법적 조치를 할 것을 엄중히 경고합니다.",
                    "이 경우, 귀하는 임대차 보증금 반환 의무에 따른 원금, 이자 및 지연 손해금은 물론, 소송비용까지 귀하가 부담하게 될 것입니다.",
                    "게다가 본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 관한 규정에 따라 연 12%가 될 것입니다.",
                    "4. 본 발신인이 귀하에게 위와 같은 법적 조치를 취하기 전에 귀하는 기한 내에 귀하의 명백한 임대차 보증금 반환 의무를 이행하여 본 건을 원만하게 해결하시기를 마지막으로 말씀드립니다."
                ]
        
        elif type == 'Yêu cầu thực hiện hợp đồng':  #계약이행 청구
            title = '계약 불이행에 대한 조치 통보'
            additional_text = [
                f"2. 본 발신인은 {information['contractDate']}일 귀하와 {information['contractName']}(이하 '본 계약')을 체결하였습니다.",
                f"이에, 귀하는 본 발신인에게 본 계약에 따라 아래와 같은 의무를 이행했어야 합니다. 그러나 귀하는 본 계약에서 정한 귀하의 의무를 이행하고 있지 않습니다.",
                f"가. 계약 내용: {information['signContent']}",
                f"나. 귀하의 의무 이행사항: {information['obligation']}",
                "3. 귀하가 본 내용증명을 받고, 위와 같은 의무를 이행하지 않을 경우 본 발신인은 귀하의 재산에 대하여 민사집행법 제276조 이하 등에 따른 가압류 등 보전 처분 및 민사 소송법 등에 따른 민사 소송은 물론 형법이나 관련 특별법 등에 따른 형사상 고소 및 행정제재 등의 법적 조치를 할 것을 엄중히 경고합니다.",
                "이 경우, 귀하는 본 발신인이 요청한 금원 및 이에 대한 지연손해금은 물론, 소송비용까지 부담하게 될 것입니다.",
                "본 발신인이 소송을 제기하면 지연손해금은 소송촉진 등에 관한 특례법 제3조 제1항 본문의 법정이율에 관한 규정에 따라 연 12%가 될 것입니다.",
                "4. 본 발신인이 귀하에게 위와 같은 법적 조치를 취하기 전에 귀하는 기한 내에 위와 같은 의무를 이행하여 본 건을 원만하게 해결하시기를 마지막으로 말씀드립니다.",
            ]
            return title, base_text[:1] + additional_text

        else:
            return ["지원되지 않는 요청 타입입니다."]        
        
    ### 2번은 로직이 잘못돼서 준비중이라고 표시
    title, result = generate_naeyong(type, information, json_input)
    if result==None:
        return jsonify({"result": '준비중입니다'})
    return jsonify({"subject" : title, "content": result})


@app.route("/translate", methods=["POST"])
def translate():
    json_input = request.get_json()
    title = json_input.get("subject","")
    naeyong = json_input.get("content", "")
    
    ###번역
    tokenizer.src_lang = 'ko'
    inputs = tokenizer(title, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id('vi'))
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    translate_title = translated_text[7:]
    
    translated = []
    for text in naeyong:
        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id('vi'))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(translated_text[7:])
    
    return jsonify({"t_subject" : translate_title, "t_content" : translated})

##커뮤니티 번역 한국-베트남(임시)
@app.route("/community_kovi", methods=["POST"])
def translate_co_kovi():
    data = request.get_json()
    text = data.get("text", "")
    to_translate = text.split('\n')
    
    source_lang = data.get("source_lang", "ko")
    target_lang = data.get("target_lang", "vi")
    
    translated=[]
    for sentence in to_translate:
        inputs = tokenizer(sentence, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(translated_text[7:])
    
    return jsonify({"content": translated})
        
##커뮤니티 번역 베트남-한국(임시)
@app.route("/community_viko", methods=["POST"])
def translate_co_viko():
    data = request.get_json()
    text = data.get("text", "")
    to_translate = text.split('\n')
    
    source_lang = data.get("source_lang", "vi")
    target_lang = data.get("target_lang", "ko")
    
    translated=[]
    for sentence in to_translate:
        inputs = tokenizer(sentence, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(target_lang))
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        translated.append(translated_text[7:])
    
    return jsonify({"content": translated})


if __name__ == "__main__":
    app.run(port=5000, debug = True)