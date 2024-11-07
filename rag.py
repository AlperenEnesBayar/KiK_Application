"""
Runs a RAG application backed by a txtai Embeddings database.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TIKA_LOG_PATH'] = '/media/alperk/Disk/KiK/KiK_Application/'
import platform
import re

from glob import glob
from io import BytesIO
from uuid import UUID

from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import pandas as pd


from txtai import Embeddings, RAG
from txtai.pipeline import Textractor
import xml.etree.ElementTree as ET
import difflib
from joblib import Parallel, delayed
from MainSearch import MainSearch, Config

from rapidfuzz import fuzz
import time


# Build logger
logger = st.logger.get_logger(__name__)


highlight_colors = {
    0: (255,159,174, 0.3),
    1: (253, 194, 149, 0.3),
    2: (166, 225, 171, 0.3),
    3: (167, 224, 246, 0.3),
    4: (244, 167, 251, 0.3)
}

highlight_colors_streamlit = {
    0: 'red',      # (255,159,174, 0.3) — close to a red-pink shade
    1: 'orange',   # (253,233,149, 0.3) — yellow-orange tone
    2: 'green',    # (166,225,197, 0.3) — a soft green shade
    3: 'blue',     # (167,224,246, 0.3) — light blue color
    4: 'violet'    # (225,167,251, 0.3) — purple/violet tone
}



# Get the article

def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, f"<span style='background-color: yellow'>{keyword}</span>")
        # text = re.sub(rf'\b{keyword}\b', f"<span style='background-color: yellow'>{keyword}</span>", text)
    return text

class AutoId:
    """
    Helper methods to detect txtai auto ids
    """

    @staticmethod
    def valid(uid):
        try:
            return UUID(str(uid))
        except ValueError:
            pass

        # Return True if this is numeric, False otherwise
        return isinstance(uid, int) or uid.isdigit()


class Application:
    """
    RAG application
    """

    def __init__(self):
        """
        Creates a new application.
        """
        model_name = Config.ModelName_BAAI_BGE_M3

        self.context = int(os.environ.get("CONTEXT", 10))
        self.main_search = MainSearch(model_name=model_name, chunk_type=Config.ChunkType_512, chunking_method=Config.ChunkingMethod_NEW)
        
        self.main_categories = {
            "Kanunlar": ["4734", "4735"],
            "Tebliğler": [
                "4734 Sayılı Kamu İhale Kanununun 62 nci Maddesinin (ı) Bendi Kapsamında Yapılacak Başvurulara İlişkin Tebliğ",
                "Doğrudan Temin Yöntemiyle Yapılacak Alımlara İlişkin Tebliğ",
                "Eşik Değerler ve Parasal Limitler Tebliği",
                "İhalelere Yönelik Başvurular Hakkında Tebliğ",
                "Kamu İhale Genel Tebliği",
                "Kamu Özel İş Birliği Projeleri ile Lisanslı İşler Kapsamında Gerçekleştirilen Yapım İşlerine İlişkin İş Deneyim Belgeleri Hakkında Tebliğ",
                "Yapım İşleri Benzer İş Grupları Tebliği"
            ],
            "Esaslar": ["Fiyat Farkı Esasları"],
            "Yönetmelikler": [
                "İhale Uygulama Yönetmelikleri",
                "Muayene ve Kabul Yönetmelikleri",
                "İhalelere Yönelik Başvurular Hakkında Yönetmelik"
            ],
            "Yönergeler": [
                "İtirazen Şikayet Başvuru Bedelinin İadesine İlişkin Yönerge",
                "Yurt Dışında Yapım İşlerinden Elde Edilen İş Deneyim Belgelerinin Belgelerin Sunuluş Şekline Uygunluğunu Tevsik Amacıyla EKAP'a Kaydedilmesine İlişkin Yönerg"
            ]
        }

        csv_file = "kik_all_madde.csv"
        self.df = pd.read_csv(csv_file)

    
    @Config.calculate_runtime
    def get_madde_new(self, file_name, maddes, maddes_full):

        start_find = time.time()
        files = self.df['FileName'].tolist()
        # uniques
        files = list(set(files))

        file_name = file_name.replace('.txt', '').replace('.pdf', '').replace('.docx', '').replace('.doc', '')
        
        found_file = None
        for file in files:
            fname = os.path.basename(file).split('.')[0]
            if file_name == fname:
                found_file = file
                break

                
        results = []
        if found_file:
            filtered_rows = self.df[self.df['FileName'] == found_file]
            filtered_rows = filtered_rows[filtered_rows['Kanun No'].isin(maddes)]
            for ash, (_, row) in enumerate(filtered_rows.iterrows()):
                results.append({
                    "Title": row["Title"],
                    "Kanun": row["Kanun"],
                    "Kanun No": row["Kanun No"],
                    "FileName": row["FileName"]
                })
        
        return results


    @Config.calculate_runtime
    def get_madde(self, text, file_name):
        start_total = time.time()
        def get_overlap(s1, s2):
            start_overlap = time.time()
            similarity_score = fuzz.partial_ratio(s1, s2, score_cutoff=70)
            end_overlap = time.time()
            
            return similarity_score
        
        def process_row(row, search_term, matches):
            madde_content = row['Kanun']
            madde_no = str(row["Kanun No"]).strip()
            if madde_no in matches:
                return row, None
            else:
                match_ratio = get_overlap(search_term, madde_content)
                if match_ratio > 60:
                    return row, match_ratio
            return None, match_ratio
    
        def search_madde_csv_parallel(filtered_rows, search_term):
            # (define 'pattern' and 'matches' as in your function)

            pattern = re.compile(r'(?i)(?<!Geçici\s)Madde\s*(\d+)')
            matches = pattern.findall(search_term)
            results_ratios = Parallel(n_jobs=-1)(
                delayed(process_row)(row, search_term, matches) for _, row in filtered_rows.iterrows()
            )

            results, ratios = zip(*results_ratios)
            results = [result for result in results if result is not None]

            return results
    
        def find_article(chunk, file_result, df):
            start_find = time.time()
            files = df['FileName'].tolist()
            # uniques
            files = list(set(files))

            file_name = file_result.replace('.txt', '').replace('.pdf', '').replace('.docx', '').replace('.doc', '')
            found_file = None

            for file in files:
                fname = os.path.basename(file).split('.')[0]
                if file_name == fname:
                    found_file = file
                    break

            if found_file:
                filtered_rows = df[df['FileName'] == found_file]

                results = search_madde_csv_parallel(filtered_rows, chunk)
                end_find = time.time()
                return results
            
            end_find = time.time()
           
            return None
        

        start_find_maddes = time.time()
        found_maddes = find_article(text, file_name, self.df)
        end_find_maddes = time.time()

        # End total runtime
        end_total = time.time()

        # Calculate and print percentages
        total_runtime = end_total - start_total

        return found_maddes


    def addurl(self, url):
        print("Not implemented")
        print("LOGAEB: ", url)

    def create(self):
        print("Not implemented")
        print("LOGAEB: ", "create")
        return self.main_search


    def stream(self, data):
        for sections in self.extract(glob(f"{data}/**/*", recursive=True)):
            yield from sections

    def extract(self, inputs):
        textractor = Textractor(paragraphs=True)
        return textractor(inputs)


    def instructions(self):
        instructions = (
            f"Kamu İhale Kurumu Akıllı Arama Motoru 🚀'na soru sormaya başlayabilirsiniz.")

        return instructions

    def settings(self):
        config = "\n".join(
            f"|{name}|{os.environ.get(name)}|"
            for name in ["EMBEDDINGS", "DATA", "PERSIST", "LLM"]
            if name
        )

        return (
            "The following is a table with the current settings.\n"
            f"|Name|Value|\n"
            f"|----|-----|\n"
            f"|RECORD COUNT|{self.embeddings.count()}|\n"
        ) + config


    def toggle_drawer(self):
        st.session_state['drawer_open'] = not st.session_state['drawer_open']


    def run(self):

        radio_button_option = None
        hybrid_param_option = None
        topk_option = None
        option = None
        option2 = None

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                radio_button_option = st.radio("Hangi versiyon kullanılsın?", [Config.ChunkingMethod_OLD, Config.ChunkingMethod_NEW],
                                               index=1,
                                            captions=["Toplantıda gösterdiğimiz version", "Son yapılan geliştirmelerin olduğu version"])
        with col2:
            with st.container(border=False):
                topk_option = st.number_input("Toplam Cevap Sayısı:", min_value=1, max_value=20, value=5, step=1)
            with st.container(border=False):
                hybrid_param_option = st.slider("Hybrid Parametresi:", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

        if radio_button_option == Config.ChunkingMethod_OLD:
        
            col1, col2 = st.columns(2)
            with col1:
                option = st.selectbox(
                    "Arama Nerede Yapılsın?",
                    (Config.Option_All, Config.Option_Kanun, Config.Option_Teblig, Config.Option_Esas, Config.Option_Yonetmelik, Config.Option_Yonerge),
                    key="main_categories-single"
                )

            if option != "Bütün Veriler":
                option_idx = list(self.main_categories.keys()).index(option)

                if option == "Kanunlar":
                    with col2:
                        option2 = st.selectbox("Kanunlar", (
                            # "Hepsi",
                            # "4734", 
                            # "4735"
                            Config.Option_Kanun_Hepsi,
                            Config.Option_Kanun_4734,
                            Config.Option_Kanun_4735
                            ))
                elif option == "Tebliğler":
                    with col2:
                        option2 = st.selectbox(
                            "Tebliğler",
                            (
                                Config.Option_Teblig_Hepsi,
                                Config.Option_Teblig_4734,
                                Config.Option_Teblig_Dogrudan,
                                Config.Option_Teblig_Esik,
                                Config.Option_Teblig_Ihalelere,
                                Config.Option_Teblig_KamuIhale,
                                Config.Option_Teblig_KamuOzel,
                                Config.Option_Teblig_Yapim
                            ))
                elif option == "Esaslar":
                    with col2:
                        option2 = st.selectbox("Esaslar", (
                            # "Hepsi"
                            Config.Option_Esas_Hepsi
                            ))
                elif option == "Yönetmelikler":
                    with col2:
                        option2 = st.selectbox("Yönetmelikler", (
                            # "Hepsi",
                            # "İhale Uygulama Yönetmelikleri",
                            # "Muayene ve Kabul Yönetmelikleri"
                            Config.Option_Yonetmelik_Hepsi,
                            Config.Option_Yonetmelik_Ihale,
                            Config.Option_Yonetmelik_Muayene,
                            Config.Option_Yonetmelik_Ihalelere
                            ))
                elif option == "Yönergeler":
                    with col2:
                        option2 = st.selectbox("Yönergeler", (
                            # "Hepsi",
                            # "İtirazen Şikayet Başvuru Bedelinin İadesine İlişkin Yönerge",
                            # "Yurt Dışında Yapım İşlerinden Elde Edilen İş Deneyim Belgelerinin Belgelerin Sunuluş Şekline Uygunluğunu Tevsik Amacıyla EKAP'a Kaydedilmesine İlişkin Yönerg"
                            Config.Option_Yonerge_Hepsi,
                            Config.Option_Yonerge_Itiraz,
                            Config.Option_Yonerge_Yurt
                            ))
                st.write(f'Arama Kümesi {option} -> {option2} olarak seçildi.')
            else:
                st.write(f'Arama Kümesi {option} olarak seçildi.')

            
    

            if "messages" not in st.session_state.keys():
                st.session_state.messages = [
                    {"role": "assistant", "content": self.instructions()}
                ]

            if question := st.chat_input("Sorunuzu sorabilirsiniz"):
                message = question
                if question.startswith("#"):
                    message = f"Upload request for _{message.split('#')[-1].strip()}_"

                st.session_state.messages.append({"role": "user", "content": message})

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] != "assistant"
            ):
                with st.chat_message("assistant"):
                    logger.debug(f"USER INPUT: {question}")

                    # Check for file upload
                    if question.startswith("#"):
                        url = question.split("#")[1].strip()
                        with st.spinner(f"Adding {url} to index"):
                            self.addurl(url)

                        response = f"Added _{url}_ to index"
                        st.write(response)

                    # Show settings
                    elif question == ":settings":
                        response = self.settings()
                        st.write(response)

                    else:
 
                        
                        rets = self.main_search.retreive(query=question,
                                    option=option,
                                    option2=option2,
                                    retrive_type=Config.RetriveType_RERANK,
                                    top_k=topk_option, 
                                    num_chunk=5, 
                                    hybrit_rate=hybrid_param_option, 
                                    chunk_score_threshold=0.4, 
                                    exp_name="exp", 
                                    verbose=False, 
                                    slow_run=False, 
                                    chat=False)
    
                        response_txts = []

                        st.markdown("### Sonuçlar:")

                        for idx, ret in enumerate(rets):
                            fileName = ret[0]
                            chunk = ret[1]
                            score = ret[2]
                            rank = idx + 1

                            found_maddes = self.get_madde(chunk, fileName)

                            madde_txt = ""
                            option_txt = f"**Arama Kümesi:** {option}\n\n" if option2 is None else f"**Arama Kümesi:** {option} -> {option2}\n\n"

                            title_txt = f"### Sonuç - {rank}\n\n"
                            response_txt = f"{option_txt} **Bulunduğu Dosya:** {fileName.replace('.txt', '')}"
                            
                            title_response_txt = f"\n\n#### Madde:\n\n"

                            madde_links = []

                            if found_maddes:
                                for idy, madde in enumerate(found_maddes):
                                    # check if title is pd nan 
                                    c_title = str(madde["Title"]).strip() if pd.isna(madde["Title"]) == False else "Madde başlığı bulunmamaktadır." 
                                    
                                    # change color of text with highlight colors
                                    # link_text =  '*'+ c_title + "* - **Madde - "+ str(madde["Kanun No"]).strip() + '**'
                                    link_text =  f':{highlight_colors_streamlit[idy%5]}[*'+ c_title + "*] - **Madde - "+ str(madde["Kanun No"]).strip() + '**'
                                    
                                    sanitized_text = madde["Kanun"]
                                    madde_links.append({
                                        "link_text": link_text,
                                        "content_text": sanitized_text
                                    })
                            else:
                                response_txt += "Madde bulunamadı."

        
                            st.markdown(title_txt, unsafe_allow_html=True)
                            container = st.container()
                            container.markdown(response_txt, unsafe_allow_html=True)

                            st.markdown(title_response_txt, unsafe_allow_html=True)
                            # st.markdown(response_txt, unsafe_allow_html=True) 
                            response_txts.append(response_txt)
                            

                            for madde_link in madde_links:
                                link_text = madde_link["link_text"]
                                content_text = madde_link["content_text"]
                                
                                with st.expander(link_text):
                                    if len(content_text.split(' ')) > 170:
                                        st.markdown(
                                            f"""
                                            <div contenteditable="true" style="resize: vertical; overflow: auto; border: 1px solid; padding: 10px; height: 200px; min-height: 200px;">
                                                {content_text}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            '<div style="text-align: justify;">' +
                                            content_text +
                                            '</div>', 
                                            unsafe_allow_html=True)
                                


                            sub_chunks = re.split(r"Madde\s([1-9][0-9]?|0)", chunk, flags=re.IGNORECASE)
                            
                            for idz, sub_chunk in enumerate(sub_chunks):
                                if len(sub_chunk) < 10:
                                    continue
                                st.markdown(
                                    f'<div style="text-align: justify; background-color: rgba{highlight_colors[idz%5]};' + 
                                    'padding: 0.2em 0.4em;' +
                                    'border-radius: 0.4em;' +
                                    'color: black;' +
                                    '">' +
                                    sub_chunk + 'Madde' +
                                    '</div>', 
                                    unsafe_allow_html=True)
                            
                            st.divider()
                            

                        st.session_state.messages.append(
                            {"role": "assistant", "content": "\n\n".join(response_txts)})
        elif radio_button_option == Config.ChunkingMethod_NEW:
            multiselect_list = [x + ' -> Bütün Veriler' for x in self.main_categories.keys()]
            multiselect_list.extend([x + ' -> ' + y for x in self.main_categories.keys() for y in self.main_categories[x]])
            multiselect_list.append("Bütün Veriler")

            multiselect_option = st.multiselect(
                "Arama Nerede Yapılsın?", 
                # [x for x in self.main_categories.keys()].extend([x for x in self.main_categories.values()]),
                multiselect_list,
                default=["Bütün Veriler"])
    

            if "messages" not in st.session_state.keys():
                st.session_state.messages = [
                    {"role": "assistant", "content": self.instructions()}
                ]

            if question := st.chat_input("Sorunuzu sorabilirsiniz"):
                message = question
                if question.startswith("#"):
                    message = f"Upload request for _{message.split('#')[-1].strip()}_"

                st.session_state.messages.append({"role": "user", "content": '(Önceki Soru) - 'message})

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if (
                st.session_state.messages
                and st.session_state.messages[-1]["role"] != "assistant"
            ):
                with st.chat_message("assistant"):
                    logger.debug(f"USER INPUT: {question}")

                    # Check for file upload
                    if question.startswith("#"):
                        url = question.split("#")[1].strip()
                        with st.spinner(f"Adding {url} to index"):
                            self.addurl(url)

                        response = f"Added _{url}_ to index"
                        st.write(response)

                    # Show settings
                    elif question == ":settings":
                        response = self.settings()
                        st.write(response)

                    else:
                        # Check for Graph RAG
                        print(multiselect_option)
                        rets = self.main_search.retreive(query=question,
                                    option=option,
                                    option2=option2,
                                    option3=multiselect_option,
                                    retrive_type=Config.RetriveType_RERANK,
                                    top_k=topk_option, 
                                    num_chunk=5, 
                                    hybrit_rate=hybrid_param_option, 
                                    chunk_score_threshold=0.4, 
                                    exp_name="exp", 
                                    verbose=False, 
                                    slow_run=False, 
                                    chat=False)
    
                        response_txts = []

                        st.markdown("### Sonuçlar:")

                        for idx, ret in enumerate(rets):
                            fileName = ret[0]
                            chunks = ret[1].split('&&')
                            score = ret[2]
                            kanunnos = ret[3].split('&&')
                            rank = idx + 1

                            search_kanun = []
                            search_kanun_full = []
                            for kanunno in kanunnos:
                                temp = kanunno.split(' - ')[0].strip()
                                if temp != '':
                                    search_kanun.append(temp)
                                    search_kanun_full.append(kanunno.strip())


                            found_maddes = self.get_madde_new(fileName, search_kanun, search_kanun_full)

                            

                            madde_txt = ""
                            option_txt = f"**Arama Kümesi:** {option}\n\n" if option2 is None else f"**Arama Kümesi:** {multiselect_option}\n\n"

                            title_txt = f"### Sonuç - {rank}\n\n"
                            response_txt = f"{option_txt} **Bulunduğu Dosya:** {fileName.replace('.txt', '')}"
                            
                            title_response_txt = f"\n\n#### Madde:\n\n"

                            madde_links = []

                            if found_maddes:
                                for idy, madde in enumerate(found_maddes):
                                    # check if title is pd nan 
                                    c_title = str(madde["Title"]).strip() if pd.isna(madde["Title"]) == False else "Madde başlığı bulunmamaktadır." 
                                    
                                    # change color of text with highlight colors
                                    # link_text =  '*'+ c_title + "* - **Madde - "+ str(madde["Kanun No"]).strip() + '**'
                                    link_text =  f':{highlight_colors_streamlit[idy%5]}[*'+ c_title + "*] - **Madde - "+ str(madde["Kanun No"]).strip() + '**'
                                    
                                    sanitized_text = madde["Kanun"]
                                    madde_links.append({
                                        "link_text": link_text,
                                        "content_text": sanitized_text
                                    })
                            else:
                                response_txt += "Madde bulunamadı."

        
                            st.markdown(title_txt, unsafe_allow_html=True)
                            container = st.container()
                            container.markdown(response_txt, unsafe_allow_html=True)

                            st.markdown(title_response_txt, unsafe_allow_html=True)
                            # st.markdown(response_txt, unsafe_allow_html=True) 
                            response_txts.append(response_txt)
                            

                            for madde_link in madde_links:
                                link_text = madde_link["link_text"]
                                content_text = madde_link["content_text"]
                                
                                with st.expander(link_text):
                                    if len(content_text.split(' ')) > 170:
                                        st.markdown(
                                            f"""
                                            <div contenteditable="true" style="resize: vertical; overflow: auto; border: 1px solid; padding: 10px; height: 200px; min-height: 200px;">
                                                {content_text}
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )
                                    else:
                                        st.markdown(
                                            '<div style="text-align: justify;">' +
                                            content_text +
                                            '</div>', 
                                            unsafe_allow_html=True)
                                


                            
                            
                            for idz, sub_chunk in enumerate(chunks):
                                if len(sub_chunk) < 10:
                                    continue
                                st.markdown(
                                    f'<div style="text-align: justify; background-color: rgba{highlight_colors[idz%5]};' + 
                                    'padding: 0.2em 0.4em;' +
                                    'border-radius: 0.4em;' +
                                    'color: black;' +
                                    '">' +
                                    '<b>Madde - ' +
                                    search_kanun_full[idz] +  '</b> - ' +
                                    self.main_search.highlight_chunk_with_lemmas_substring(sub_chunk, question) +
                                    '</div>', 
                                    unsafe_allow_html=True)
                            
                            st.divider()
                            

                        # st.session_state.messages.append(
                        #     {"role": "assistant", "content": "\n\n".join(response_txts)})


@st.cache_resource(show_spinner="Modeller yükleniyor... Veri setleri oluşturuluyor...")
def create():
    """
    Creates and caches a Streamlit application.

    Returns:
        Application
    """

    return Application()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    st.set_page_config(
        page_title="Kamu İhale Kurumu - MAIN Akıllı Arama",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title(os.environ.get("TITLE", "Kamu İhale Kurumu - MAIN Akıllı Arama"))


    print("Starting RAG application...")
    app = create()
    print("RAG application started.")
    app.run()
    print("RAG application finished.")



yorum = """
print("")
chunk_text = ""
for f_madde in found_maddes:
    # find overlap between chunk and madde
    madde_content = f_madde['Kanun']
    madde_no = str(f_madde["Kanun No"]).strip()
    m_match = SequenceMatcher(None, chunk, madde_content).find_longest_match(0, len(chunk), 0, len(madde_content))
    if m_match.size > 0:
        overlap = chunk[m_match.a: m_match.a + m_match.size]
        temp_text = f'<div style="text-align: justify; background-color: rgba{highlight_colors[idz%5]};'
        temp_text += 'padding: 0.2em 0.4em;'
        temp_text += 'border-radius: 0.4em;'
        temp_text += 'color: black;'
        temp_text += '">'
        temp_text += ''
        '</div>',

"""
