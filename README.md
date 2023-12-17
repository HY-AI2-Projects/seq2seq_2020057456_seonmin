# seq2seq_2020057456_seonmin
● seq2seq_2020057456_seonmin_기말과제수행 <br>
● 본 페이지는 학교 기말 과제 수행을 위한 논문 리뷰 게시글입니다. <br>
● 논문 링크 : https://arxiv.org/abs/1409.3215 <br>

# ● RNN(순환신경망,Recurrent Neural Network) <br>
![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/da1da5ef-e80f-4e01-8bd6-decb57f46f69)
 RNN은 순환 신경망으로 입력 레이어, 히든 레이어, 출력 레이어로 구성된 아키텍쳐를 가지고 있습니다. 해당 그림은 RNN의 구조를 간단하게 보여준 것인데요. 보시는 것과 같이 특정 시간 t에 입력된 xt를 처리한 뒤에 출력 ht를 내보내며, 이 다음 노드로만 계산되는 것이 아니라 자기 자신에게 혹은 그 전 노드에 대한 정보들을 함께 받아 다음 노드로 계산되는 구조를 볼 수 있습니다. 이처럼 다음 시점을 계산하기 위해 전 시점의 정보를 저장하는 구조로 다음 시점의 정보는 전 시점의 정보만이 아니라 이전까지의 정보들을 모두 가지고 있을 것입니다. 이러한 정보를 가지고 있는 것을 cell이라고 하며 현재 cell이 가지고 있는 정보, 즉 다음 시점으로 넘겨줄 정보를 hidden state라고 이야기 합니다.

# ● seq2seq 원리 <br>
 ![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/8f740bf3-c365-492f-b2a3-737d7b077d24)
 seq2seq 구조로 인코더를 통해 A,B,C라는 단어들을 입력 받고 LSTM을 통해서 인코더에서 여러번 hidden state 값들이 갱신이 되고 이제 EOS 토큰으로 끝나는데 입출력 문장 모두 해당 토큰으로 문장의 끝을 알 수 있습니다. 결과적으로 인코더 파트에서 나오게 된 hidden state 값을 context vector를 거쳐 이제 인코더의 마지막 hidden state만을 고정된 크기의 context vector로 정보를 전달하여 디코더 파트에서 히든 스테이트에 대한 정보를 통해 W,X,Y,Z라는 번역 결과를 출력합니다.<br>
 이러한 LSTM를 이용해서 시퀀스 2 시퀀스 모델을 이용하는 것 뿐만 아니라 학습을 진행하는 과정에서 입력 문장의 순서를 바꾸는 것이 더욱 성능을 비약적으로 향상시켰다고 논문에서 밝히고 있습니다. 그림에서는 A,B,C 순서로 데이터가 들어가는 것처럼 그림이 그려져 있지만 실제로 구현을 할 때에는 A,B,C 의 순서를 뒤바꿔서 C,B,A로 구현을 하였고 평가할 때도 마찬가지로 데이터를 뒤집어서 만드는 것이 성능이 더 좋았다고 평가하였습니다.

![image](https://github.com/HY-AI2-Projects/seq2seq_2020057456_seonmin/assets/153084250/4d74db0a-5b8f-4841-97fd-2d2e04d12a56)
 논문에서 제시 되었던 seq2seq에서는 start of sequence가 없었던 이유는 입력 문장 x는 여러개의 단어로 이루어져 있는데 이때 입력 문장의 단어의 개수와 출력 문자의 단어의 개수가 같을 필요가 없다고 합니다. 입력 문제는 하나의 context vector인 v로 바뀌게 돼고 이 v를 이용해서 매번 출력 결과를 디코더 파트를 거쳐서 결과 문장이 나올 수 있도록 만들 수 있습니다. 그래서 현 논문에서는 별도의 start of sequence가 없다고 가정하고 전개되었습니다.

# ● seq2seq 코드 구현

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, verbose=0)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0
        )

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence




# ● 출처
그림1 : https://ctkim.tistory.com/
코드 구현 : https://www.kaggle.com/code/kmkarakaya/part-a-introduction-to-seq2seq-learning?cellIds=12&kernelSessionId=47250687
