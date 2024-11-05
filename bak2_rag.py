"""
Runs a RAG application backed by a txtai Embeddings database.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TIKA_LOG_PATH'] = '/data/Workspace/aebayar/KiK_RaG/KiK_Application/'
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
        self.index_file = '/data/Workspace/aebayar/KiK_RaG/Chat_GPT_Indexes/bKik_index_model_bge-m3_01-10-2024'
        model_name = Config.ModelName_BAAI_BGE_M3
        csv_file = 'Data/kik_512_chunk_17_09_24.csv'

        self.context = int(os.environ.get("CONTEXT", 10))
        self.main_search = MainSearch(model_name=model_name, indexed_csv_file=csv_file, chunk_type=Config.ChunkType_512)
        
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


    
    def get_madde(self, text, file_name):
        start_total = time.time()
        def get_overlap(s1, s2):
            start_overlap = time.time()
            similarity_score = fuzz.partial_ratio(s1, s2, score_cutoff=70)
            end_overlap = time.time()
            print(f"get_overlap runtime: {end_overlap - start_overlap:.4f} seconds")
            
            return similarity_score
        
        def process_row(row, search_term, matches):
            madde_content = row['Kanun']
            madde_no = str(row["Kanun No"]).strip()
            if madde_no in matches:
                return row, None
            else:
                match_ratio = get_overlap(search_term, madde_content)
                if match_ratio > 70:
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
        
        def search_madde_csv(filtered_rows, search_term):
            start_search = time.time()
            results = []
            ratios = []
            
            pattern = re.compile(r'(?i)Madde\s*(\d+)')
            matches = pattern.findall(search_term)
            print("LOGAEB matches: ", matches)

            for index, row in filtered_rows.iterrows():
                madde_content = row['Kanun']
                
                madde_no = str(row["Kanun No"]).strip()
                if madde_no in matches:
                    results.append(row)
                else:
                    match_ratio = get_overlap(search_term, madde_content)
                    ratios.append(match_ratio)
                    if  match_ratio > 70:  
                        results.append(row)
            
            end_search = time.time()
            print(f"search_madde_csv total runtime: {end_search - start_search:.4f} seconds")
           
            return results

        def find_article(chunk, file_result, df):
            start_find = time.time()
            files = df['FileName'].tolist()
            file_name = file_result.split('.')[0]
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
                print(f"find_article runtime (file not found case): {end_find - start_find:.4f} seconds")
                return results
            
            end_find = time.time()
            print(f"find_article runtime (file not found case): {end_find - start_find:.4f} seconds")
           
            return None
        

        start_find_maddes = time.time()
        found_maddes = find_article(text, file_name, self.df)
        end_find_maddes = time.time()

        # End total runtime
        end_total = time.time()

        # Calculate and print percentages
        total_runtime = end_total - start_total
        print(f"Total runtime: {total_runtime:.4f} seconds")
        print(f"find_article runtime: {(end_find_maddes - start_find_maddes) / total_runtime * 100:.2f}% of total time")
        print(f"get_madde runtime (excluding find_article): {(total_runtime - (end_find_maddes - start_find_maddes)) / total_runtime * 100:.2f}% of total time")

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
        option = None
        option2 = None
        
        col1, col2 = st.columns(2)
        with col1:
            option = st.selectbox(
                "Arama Nerede Yapılsın?",
                (Config.Option_All, Config.Option_Kanun, Config.Option_Teblig, Config.Option_Esas, Config.Option_Yonetmelik, Config.Option_Yonerge),
                key="main_categories-single"
            )

        if option != "Bütün Veriler":
            option_idx = list(self.main_categories.keys()).index(option)
            print("LOGAEB: ", "option_idx", option_idx)
            print("LOGAEB: ", "option", option)
        
            
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
                    # Check for Graph RAG
                    print("LOGAEB: ", question)
                    
                    rets = self.main_search.retreive(query=question,
                                option=option,
                                option2=option2,
                                retrive_type=Config.RetriveType_RERANK,
                                top_k=5, 
                                num_chunk=50, 
                                hybrit_rate=0.5, 
                                chunk_score_threshold=0.4, 
                                exp_name="exp", 
                                verbose=False, 
                                slow_run=False, 
                                chat=False)
  


                    response_txts = []

                    st.markdown("## Sonuçlar:")

                    for idx, ret in enumerate(rets):
                        fileName = ret[0]
                        chunk = ret[1]
                        score = ret[2]
                        rank = idx + 1

                        found_maddes = self.get_madde(chunk, fileName)
                        # found_maddes = [
                        #     {"Title": "Madde", "Kanun No": "4734", "Kanun": "Madde içeriği"}, 
                        #     {"Title": "Madde2", "Kanun No": "47342", "Kanun": "Madde içeriği2"}
                        # ]
                        

                        madde_txt = ""
                        option_txt = f"**Arama Kümesi:** {option}\n\n" if option2 is None else f"**Arama Kümesi:** {option} -> {option2}\n\n"
                        
                        title_txt = f"### Sonuç-{rank}\n\n"
                        response_txt = f"{option_txt} **Bulunduğu Dosya:** {fileName.replace('.txt', '')}"
                        
                        title_response_txt = f"\n\n#### Madde:\n\n"

                        madde_links = []

                        if found_maddes:
                            for idy, madde in enumerate(found_maddes):
                                # check if title is pd nan 
                                c_title = str(madde["Title"]).strip() if pd.isna(madde["Title"]) == False else "Madde başlığı bulunmamaktadır." 
                                link_text =  '*'+ c_title + "* - **Madde - "+ str(madde["Kanun No"]).strip() + '**'
                                sanitized_text = madde["Kanun"]
                                madde_links.append({
                                    "link_text": link_text,
                                    "content_text": sanitized_text
                                })
                        else:
                            response_txt += "Madde bulunamadı."

                        title_txt = """
                        <div style="font-size: 24px; font-weight: bold; color: #4A90E2;">
                        """ + title_txt + """
                        <span style="font-style: italic; color: #FF5722;">Özel Mesajınız</span>
                        </div>
                        """

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
                            

                        st.markdown(
                            '<div style="text-align: justify;">' +
                            chunk +
                            '</div>', 
                            unsafe_allow_html=True)

                        st.divider()

                    st.session_state.messages.append(
                        {"role": "assistant", "content": "\n\n".join(response_txts)})



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
        page_title="MAIN - Kamu İhale Kurumu - Akıllı Arama Motoru",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title(os.environ.get("TITLE", "MAIN - Kamu İhale Kurumu - Akıllı Arama Motoru 🚀"))


    print("Starting RAG application...")
    app = create()
    print("RAG application started.")
    app.run()
    print("RAG application finished.")
