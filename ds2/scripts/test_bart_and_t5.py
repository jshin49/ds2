from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


def test_bart_and_t5():
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    t5_tokenizer = AutoTokenizer.from_pretrained("t5-large")

    bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")

    # for t5, decoder start token == pad token
    print(t5_model.config.decoder_start_token_id, t5_tokenizer.eos_token_id, t5_tokenizer.pad_token_id)
    # for bart, decoder start token == eos token
    print(bart_model.config.decoder_start_token_id, bart_tokenizer.bos_token_id, bart_tokenizer.eos_token_id, bart_tokenizer.pad_token_id)

    conv_text = " ".join([
        "system: none",
        "user: can i get a train to cambridge on friday ?",
        "system: there is a depature at london kings cross at 7:17 am .",
        "user: actually , i need to depart from leicester after 12:15 .",
        "system: yes , there is a train that departs at 13:09 , arriving in cambridge at 14:54 . would that work for you ?",
        "user: yes . could you give me the train id ?",
    ]).lower()

    conv_summ = "The user is looking for a train that leaves at 12:15 from cambridge to leicester on friday.".lower()
    print("summ", conv_summ)
    print("t5 labels", t5_tokenizer(conv_summ, return_attention_mask=False).input_ids)
    print("t5 decode labels", t5_tokenizer.decode(t5_tokenizer(conv_summ).input_ids))
    print("t5 inputs", t5_model.prepare_decoder_input_ids_from_labels(t5_tokenizer(conv_summ, return_tensors="pt").input_ids))
    # print("t5 decode inputs", t5_tokenizer.decode(t5_model.prepare_decoder_input_ids_from_labels(t5_tokenizer(conv_summ, return_tensors="pt").input_ids)).tolist()[0])
    print("bart labels", bart_tokenizer(conv_summ).input_ids)
    print("bart decode labels", bart_tokenizer.decode(bart_tokenizer(conv_summ).input_ids))
    print("bart inputs", bart_model.prepare_decoder_input_ids_from_labels(bart_tokenizer(conv_summ, return_tensors="pt").input_ids))
    # print("bart decode inputs", bart_tokenizer.decode(bart_model.prepare_decoder_input_ids_from_labels(bart_tokenizer(conv_summ, return_tensors="pt").input_ids)).tolist()[0])

    t5_encoder_input_ids = t5_tokenizer(f"summarize: {conv_text}", return_tensors="pt").input_ids
    t5_encoder_attention_mask = t5_tokenizer(f"summarize: {conv_text}", return_tensors="pt").attention_mask
    t5_labels = t5_tokenizer(conv_summ, return_tensors="pt").input_ids
    
    bart_encoder_input_ids = bart_tokenizer(conv_text, return_tensors="pt").input_ids
    bart_encoder_attention_mask = bart_tokenizer(conv_text, return_tensors="pt").attention_mask
    bart_labels = bart_tokenizer(conv_summ, return_tensors="pt").input_ids

    t5_outputs = t5_model(
        input_ids=t5_encoder_input_ids,
        attention_mask=t5_encoder_attention_mask,
        labels=t5_labels,
    )
    bart_outputs = bart_model(
        input_ids=bart_encoder_input_ids,
        attention_mask=bart_encoder_attention_mask,
        labels=bart_labels,
    )

    print(t5_outputs.loss, bart_outputs.loss)

    t5_labels.masked_fill_(t5_labels == -100, t5_tokenizer.pad_token_id)
    bart_labels.masked_fill_(bart_labels == -100, bart_tokenizer.pad_token_id)
    t5_outputs = t5_model(
        input_ids=t5_encoder_input_ids,
        attention_mask=t5_encoder_attention_mask,
        labels=t5_labels,
    )
    bart_outputs = bart_model(
        input_ids=bart_encoder_input_ids,
        attention_mask=bart_encoder_attention_mask,
        labels=bart_labels,
    )

    print(t5_outputs.loss, bart_outputs.loss)

    # t5_encoder_inputs = t5_tokenizer(conv_text, return_tensors="pt")
    # t5_input_ids = t5_encoder_inputs["input_ids"]
    # t5_attention_mask = t5_encoder_inputs["attention_mask"]
    # t5_labels = t5_tokenizer(conv_summ, return_tensors="pt", add_special_tokens=False, return_attention_mask=False)
    
    # input_text = "The quick brown fox jumps over the lazy dog"
    # input_ids = t5_tokenizer.encode(input_text, return_tensors="pt")

    # output_t5 = t5_model(input_ids)
    # output_bart = bart_model(input_ids)

if __name__ == "__main__":
    test_bart_and_t5()