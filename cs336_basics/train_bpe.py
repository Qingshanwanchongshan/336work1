from collections import defaultdict
import regex as re

GPT2_SPLIT_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_pretoken_frequencies(
    text:str,special_tokens:list[str]
)->dict[tuple[bytes,...],int]:
    
    counts=defaultdict(int)
    compiled_regex=re.compile(GPT2_SPLIT_REGEX)

    if special_tokens:
        escaped_special_tokens=[re.escape(token) for token in special_tokens]
        special_pattern=re.compile("|".join(escaped_special_tokens))
        segements =special_pattern.split(text)
    else:
        segements=[text]
    
    for segement in segements:
        if not segement:
            continue

        for match in compiled_regex.finditer(segement):
            pretokem_str=match.group()

            pretokem_bytes=tuple(
                bytes([b]) for b in pretokem_str.encode("UTF-8")
            )

            counts[pretokem_bytes]+=1
    
    return counts

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
)->tuple[dict[int ,bytes],list[tuple[bytes,bytes]]]:
    vocab : dict[int ,bytes]={i:bytes([i]) for i in range(256)}
    next_id=256

    for st in special_tokens :
        st_bytes=st.encode("utf-8")
        if st_bytes not in vocab.values():
            vocab[next_id]=st_bytes
            next_id+=1
    
    with open (input_path,"r",encoding="utf-8")as f:
        text= f.read()
    
    word_counts=get_pretoken_frequencies(text,special_tokens)

    pair_counts =defaultdict(int)

    for word,freq in word_counts.items():
        for i in range(len(word)-1):
            pair =(word[i],word[i+1])
            pair_counts[pair]+=freq
    
    merges: list[tuple[bytes,bytes]]=[]
    num_merges_target= vocab_size-len(vocab)

    for _ in range (num_merges_target):
        if not pair_counts:
            break

        best_pair=max(pair_counts.keys(),key=lambda pair :(pair_counts[pair],pair))

        if pair_counts[best_pair] <= 0:
            break
        
        merges.append(best_pair)
        vocab[next_id]=best_pair[0]+best_pair[1]
        next_id +=1

        new_word_counts = {}

        for word,freq in word_counts.items():
            if len(word)<2:
                new_word_counts[word]=freq
                continue
            
            has_target =False
            for i in range (len(word)-1):
                if word[i]==best_pair[0]and word[i+1]==best_pair[1]:
                    has_target=True
                    break
            
            if not has_target:
                new_word_counts[word]=freq
                continue
            
            for i in range(len(word)-1):
                pair_counts[word[i],word[i+1]]-=freq
            
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            new_word_counts[new_word_tuple] = freq

            for i in range(len(new_word_tuple) - 1):
                pair_counts[(new_word_tuple[i], new_word_tuple[i + 1])] += freq
        word_counts = new_word_counts
        pair_counts.pop(best_pair, None)

    return vocab, merges





