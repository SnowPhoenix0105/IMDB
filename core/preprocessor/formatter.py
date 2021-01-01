import os
import email
import email.policy
import email.parser
import email.message
import re
import nltk
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter
from .log import log_message

class ParserToVectorBeforeFitException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Formatter:
    """
    vector_dimension: 向量的维数.
    symbol_set: 符号集, 该集合内的符号会在分词之前被删除, 默认为 '?', ',', '.', '!', ':', ';', '/', '\\', '\"', '\'' 
    exclude_word: 排除的词汇, 该列表内的所有词汇不会被加入到词汇表.
    vocabulary： 词汇表, 长度为vector_dimension-1.
    dictionary: key->word, \\forall word, word in vocabulary; value->word在vocabulary中的下标.

    vocabulary和dictionary通过fit()方法来初始化.

    简记vlcabulary为V, vector_dimension为D, 向量的格式:
    ( url数量, V[0], V[1], ..., V[D-2])
    """
    def __init__(self, 
        vector_dimension: "int > 1" = 10, 
        exclude_voacbulary: list = None, 
        symbol_set: set = None
        ):
        super().__init__()
        self.vocabulary = []    # 词汇表
        self.dictionary = {}    # unordered_map<str, int>
        self.exclude_vocabulary = [] if exclude_voacbulary is None \
                else [word.lower() for word in exclude_voacbulary] # 排除的词汇列表
        self.vector_dimension = vector_dimension  # 向量的维数
        self.symbol_set = symbol_set    \
            if symbol_set is not None else {
                '?', ',', '.', '!', ':', ';', '/', '\\', '\"', '\''
                } # 符号集
        self.url_re = re.compile(r"http[:/\\a-zA-Z1-9\?\&\.\=\%]*")

    def fit(self, plains: "Iterator<str>"):    
        """
        @ param plains: Iterator<str>, a list of plain
        @ return: self
        """
        log_message("@Formatter:\t start fitting...")
        counter = Counter()
        for plain in plains:
            plain_count, _ = self._plain_to_counter(plain)
            counter.update(plain_count)
        for exclude_word in self.exclude_vocabulary:
            del counter[exclude_word]
        for word, _ in counter.most_common(self.vector_dimension - 1):
            if word not in self.exclude_vocabulary:
                self.dictionary[word] = len(self.vocabulary)
                self.vocabulary.append(word)
        log_message("@Formatter:\t finish fitting!")
        log_message("@Formatter:\t vocabulary:\t" + str(self.vocabulary))
        return self

    def _plain_to_counter(self, plain: str) -> (Counter, int):
        if plain is None:
            return Counter(), 0
        plain = plain.lower()
        for symbol in self.symbol_set:
            plain = plain.replace(symbol, ' ')      # 删除标点符号
        stemmer = nltk.PorterStemmer()
        counter = Counter(map(stemmer.stem, plain.split()))
        return counter

    def all_plains_to_vectors(self, plains: "Iterable<str>") -> list:
        """
        @ param plains: 可以迭代访问的一系列简单文本
        @ return: 这些简单文本转换为向量后的列表, 列表的顺序与迭代顺序相同
        """
        return [self.plain_to_vector(plain) for plain in plains]

    def plain_to_vector(self, plain: str) -> np.ndarray:
        """
        @ param plain: 需要转换为向量的单个简单文本
        @ return: 长度为vector_dimension的一维数组
        """
        if len(self.vocabulary) == 0:
            raise ParserToVectorBeforeFitException
        counter = self._plain_to_counter(plain)
        ret = np.zeros(self.vector_dimension)
        for i in range(self.vector_dimension - 1):
            ret[i] = counter[self.vocabulary[i]]
        return ret

    @staticmethod
    def remove_html(html_content: str) -> str:
        """
        将html格式化的字符串转化为简单文本
        @ param html_content: html格式化字符串
        @ return: 转换后的简单文本, 转换失败时返回\"empty\"
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.text.replace('\n\n','')
        except Exception:
            return "empty"

    @staticmethod
    def email_message_to_plain(em: email.message.EmailMessage) -> str:
        """
        获取EmailMessage对象中的简单文本
        @ param em: EmailMessage对象
        @ return: 给定对象中的简单文本
        """
        for part in em.walk():
            part_content_type = part.get_content_type()
            if part_content_type not in ['text/plain','text/html']:
                continue
            try:
                part_content = part.get_content()
            except Exception:
                part_content = str(part.get_payload())
            if part_content_type == 'text/plain':
                return part_content if part_content is not None else "empty"
            else:
                return Formatter.html_to_plain(part_content)

    def email_message_to_vector(self, em: email.message.EmailMessage) -> tuple:
        """
        将EmailMessage对象转换为向量
        @ param em: EmailMessage对象
        @ return: 给定对象中的简单文本转换为的向量
        """
        plain = Formatter.email_message_to_plain(em)
        return self.plain_to_vector(plain)


if __name__ == "__main__":
    test_case_1 = "https://abaaba fuck \nfucking\t fucked, shit,   ooooo! greate!  see also: http://bububu"
    counter, url_count = Formatter()._plain_to_counter(test_case_1)
    print("counter = ", counter)
    print("url-count = ", url_count)
    test_case_2 = [
        "https://abaaba 11223 fuck \nfucking\t fucked, shit,  11223 https://abaaba ooooo! greate!  see also: http://bububu",
        "https://abaaba fuck 11223https://abaaba, shit,  https://abaaba ooooo! 11223greate!  see also: http://bububu",
        "https://abaaba , shit, 11223 https://abaaba ooooo!11223 greate!11223 https://abaaba see also: http://bububu"
    ]
    fmt = Formatter(vector_dimension=4, exclude_voacbulary=["11223"]).fit(test_case_2)
    vec1 = fmt.plain_to_vector(test_case_1)
    print(vec1)

