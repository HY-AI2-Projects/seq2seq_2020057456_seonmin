# seq2seq_2020057456_seonmin
● seq2seq_2020057456_seonmin_기말과제수행 <br>
● 본 페이지는 학교 기말 과제 수행을 위한 논문 리뷰 게시글입니다. <br>
● 논문 링크 : https://arxiv.org/abs/1409.3215 <br>

# ● 앞서 설명하기 <br>
 "Sequence to Sequence Learning with Neural Networks"이라는 논문으로, Ilya Sutskever, Oriol Vinyals, and Quoc V. Le에 의해 2014년에 발표되었습니다. 이 논문에서 제안된 Seq2Seq 모델은 주로 기계 번역 작업에 활용되며, 입력 시퀀스를 고정된 크기의 벡터로 인코딩하고, 이 벡터를 통해 출력 시퀀스를 디코딩하는 구조를 제안합니다.
여기에서는 Seq2Seq 논문의 핵심 아이디어를 간단하게 설명하겠습니다.

# ● RNN(순환신경망,Recurrent Neural Network) <br>
![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/da1da5ef-e80f-4e01-8bd6-decb57f46f69)
 RNN은 순환 신경망으로 입력 레이어, 히든 레이어, 출력 레이어로 구성된 아키텍쳐를 가지고 있습니다. 해당 그림은 RNN의 구조를 간단하게 보여준 것인데요. 보시는 것과 같이 특정 시간 t에 입력된 xt를 처리한 뒤에 출력 ht를 내보내며, 이 다음 노드로만 계산되는 것이 아니라 자기 자신에게 혹은 그 전 노드에 대한 정보들을 함께 받아 다음 노드로 계산되는 구조를 볼 수 있습니다. 이처럼 다음 시점을 계산하기 위해 전 시점의 정보를 저장하는 구조로 다음 시점의 정보는 전 시점의 정보만이 아니라 이전까지의 정보들을 모두 가지고 있을 것입니다. 이러한 정보를 가지고 있는 것을 cell이라고 하며 현재 cell이 가지고 있는 정보, 즉 다음 시점으로 넘겨줄 정보를 hidden state라고 이야기 합니다.

# ● seq2seq 원리 <br>
 ![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/8f740bf3-c365-492f-b2a3-737d7b077d24)
 seq2seq 구조로 인코더를 통해 A,B,C라는 단어들을 입력 받고 LSTM을 통해서 인코더에서 여러번 hidden state 값들이 갱신이 되고 이제 EOS 토큰으로 끝나는데 입출력 문장 모두 해당 토큰으로 문장의 끝을 알 수 있습니다. 결과적으로 인코더 파트에서 나오게 된 hidden state 값을 context vector를 거쳐 이제 인코더의 마지막 hidden state만을 고정된 크기의 context vector로 정보를 전달하여 디코더 파트에서 히든 스테이트에 대한 정보를 통해 W,X,Y,Z라는 번역 결과를 출력합니다.<br>
 이러한 LSTM를 이용해서 시퀀스 2 시퀀스 모델을 이용하는 것 뿐만 아니라 학습을 진행하는 과정에서 입력 문장의 순서를 바꾸는 것이 더욱 성능을 비약적으로 향상시켰다고 논문에서 밝히고 있습니다. 그림에서는 A,B,C 순서로 데이터가 들어가는 것처럼 그림이 그려져 있지만 실제로 구현을 할 때에는 A,B,C 의 순서를 뒤바꿔서 C,B,A로 구현을 하였고 평가할 때도 마찬가지로 데이터를 뒤집어서 만드는 것이 성능이 더 좋았다고 평가하였습니다.

![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/4d74db0a-5b8f-4841-97fd-2d2e04d12a56)
 논문에서 제시 되었던 seq2seq에서는 start of sequence가 없었던 이유는 입력 문장 x는 여러개의 단어로 이루어져 있는데 이때 입력 문장의 단어의 개수와 출력 문자의 단어의 개수가 같을 필요가 없다고 합니다. 입력 문제는 하나의 context vector인 v로 바뀌게 돼고 이 v를 이용해서 매번 출력 결과를 디코더 파트를 거쳐서 결과 문장이 나올 수 있도록 만들 수 있습니다. 그래서 현 논문에서는 별도의 start of sequence가 없다고 가정하고 전개되었습니다.

# ● seq2seq 핵심 아이디어 <br>
동기 (Motivation): 기존의 기계 번역 모델들은 입력과 출력의 길이를 고정시켜야 했습니다. 하지만 실제로 언어는 다양한 길이의 문장으로 이루어져 있습니다. Seq2Seq는 이러한 다양한 길이의 시퀀스에 대해 유연하게 대처할 수 있는 모델을 제안합니다.

인코더-디코더 (Encoder-Decoder) 구조: Seq2Seq 모델은 크게 인코더와 디코더로 나누어집니다. 인코더는 입력 시퀀스를 고정된 크기의 벡터로 인코딩하고, 디코더는 이 벡터를 기반으로 출력 시퀀스를 생성합니다.

LSTM (Long Short-Term Memory) 사용: 논문에서는 LSTM을 사용하여 시퀀스 정보를 처리합니다. LSTM은 긴 시퀀스에 대한 정보를 효과적으로 학습할 수 있는 RNN의 변형입니다.

Teacher Forcing: 훈련 시에는 "Teacher Forcing"이라는 방법을 사용하여 디코더의 입력으로 이전 시점의 실제 출력을 사용합니다. 이는 훈련을 안정적으로 진행시키는 데 도움이 됩니다.

Attention Mechanism (관심 메커니즘): 나중에 등장한 발전된 Seq2Seq 모델에서는 Attention Mechanism이라 불리는 기술이 추가되었습니다. 이는 디코더가 출력을 생성할 때 인코더의 다양한 부분에 "주의"를 기울일 수 있게 해줍니다.

Seq2Seq 논문은 이후의 자연어 처리 분야에서 다양한 응용에 큰 영향을 미쳤으며, 이를 토대로 여러 발전된 모델이 등장하게 되었습니다.


# ● seq2seq 코드 구현

    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense

# 데이터 준비
    input_texts = ["hello", "how are you", "goodbye"]
    target_texts = ["안녕", "잘 지내니", "안녕히 가세요"]

    input_characters = set(" ".join(input_texts))
    target_characters = set(" ".join(target_texts))

    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)

    max_encoder_seq_length = max(len(txt) for txt in input_texts)
    max_decoder_seq_length = max(len(txt) for txt in target_texts)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")
    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# 모델 정의
    latent_dim = 256

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 모델 컴파일 및 훈련
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=100, batch_size=1)

# 예측을 위한 모델 정의
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# 예측
    def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        states_value = [h, c]

    return decoded_sentence

# 예측 테스트
    for seq_index in range(len(input_texts)):
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)



# ● 출처
그림1 : https://ctkim.tistory.com/ <br>
코드 구현 : https://www.kaggle.com/code/kmkarakaya/part-a-introduction-to-seq2seq-learning?cellIds=12&kernelSessionId=47250687
