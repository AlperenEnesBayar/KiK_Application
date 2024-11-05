"""
Runs a RAG application backed by a txtai Embeddings database.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from MainSearch import MainSearch, Config

from rapidfuzz import fuzz



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
        """
        Checks if uid is a valid auto id (UUID or numeric id).

        Args:
            uid: input id

        Returns:
            True if this is an autoid, False otherwise
        """

        # Check if this is a UUID
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


    
    def get_madde(self, text, file_name):

        

        def get_overlap(s1, s2):
            similarity_score = fuzz.partial_ratio(s1, s2)
            return similarity_score
        
        def search_madde_csv(filtered_rows, search_term):

            results = []
            ratios = []
            # Iterate over all 'Madde' elements in the XML
            for index, row in filtered_rows.iterrows():
                madde_content = row['Kanun']

                match_ratio = get_overlap(search_term, madde_content)
                ratios.append(match_ratio)
                # Check if the search term is found in either the title or content
                #if search_term.split()[0] in madde_content:
                if  match_ratio > 80:  
                    results.append(row)

            print("LOGAEB: ", ratios)
            return results

        def find_article(chunk, file_result):
            #directory = '/data/Workspace/uusenturk/Akıllı Arama Mevzuat_v2_xml/'

            # Use glob to recursively find all .xml files
            #xml_files = glob(os.path.join(directory, '**', '*.xml'), recursive=True)

            csv_file = "kik_all_madde.csv"
            df = pd.read_csv(csv_file)
            files = df['FileName'].tolist()
            file_name = file_result.split('.')[0]

            found_file = None
            # Print the list of XML files
            for file in files:
                fname = os.path.basename(file).split('.')[0]
                if file_name == fname:
                    found_file = file
                    break

            if found_file:
                filtered_rows = df[df['FileName'] == found_file]

                results = search_madde_csv(filtered_rows, chunk)
                return results
            return None
        
        # Get the article
        found_maddes = find_article(text, file_name)
        # return "\n\n".join([str(madde['Title']) + " - Madde - "+ str(madde['Kanun No']) +': ' + str(madde['Kanun'])[:20] for madde in found_maddes]) if found_maddes else "Madde bulunamadı."
        return found_maddes

    def search_madde_via_embedding(self, xml_path, search_term, topk=5):
        def clear_text(text: str) -> str:
            text = re.sub(r"\n", "", text)
            text = re.sub(r"[_\[\]\{\}]", " ", text)
            text = re.sub(r"\S*\.{2,}", "", text)
            text = re.sub(r"\.{2,}\s*", " ", text)
            text = re.sub(r"\…", "", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        
        chunk = clear_text(search_term)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        results = []
        
        articles = []
        embedding = self.main_search.embedding
        # Iterate over all 'Madde' elements in the XML
        for madde in root.findall('Madde'):
            madde_content = madde.find('Content').text
            madde_content_cleared = clear_text(madde_content)
            articles.append(madde_content_cleared)
        
        uid = embedding.similarity(chunk, articles)

        return results
        
    def get_madde_via_embedding(self, text, file_result):
        directory = '/data/Workspace/uusenturk/Akıllı Arama Mevzuat_v2_xml/'

        # Use glob to recursively find all .xml files
        xml_files = glob(os.path.join(directory, '**', '*.xml'), recursive=True)

        file_name = file_result.split('.')[0]

        found_file = None
        # Print the list of XML files
        for file in xml_files:
            fname = os.path.basename(file).split('.')[0]
            if file_name == fname:
                found_file = file
                break

        if found_file:
            results = self.search_madde_via_embedding(found_file, text)
            return results
        return None


    def addurl(self, url):
        print("Not implemented")
        print("LOGAEB: ", url)

    def create(self):
        print("Not implemented")
        print("LOGAEB: ", "create")
        return self.main_search


    def stream(self, data):
        """
        Runs a textractor pipeline and streams extracted content from a data directory.

        Args:
            data: input data directory
        """

        # Stream sections from content
        for sections in self.extract(glob(f"{data}/**/*", recursive=True)):
            yield from sections

    def extract(self, inputs):
        """
        Extract sections from inputs using a Textractor pipeline.

        Args:
            inputs: input content

        Returns:
            extracted content
        """

        textractor = Textractor(paragraphs=True)
        return textractor(inputs)


    def instructions(self):
        """
        Generates a welcome message with instructions.

        Returns:
            instructions
        """

        # Base instructions
        instructions = (
            f"Kamu İhale Kurumu Akıllı Arama Motoru 🚀'na soru sormaya başlayabilirsiniz.")

        return instructions

    def settings(self):
        """
        Generates a message with current settings.

        Returns:
            settings
        """

        # Generate config settings rows
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
        """
        Runs a Streamlit application.
        """
        # Create two columns with st.columns

        if 'drawer_open' not in st.session_state:
            st.session_state['drawer_open'] = False

        option = None
        option2 = None
        
        col1, col2 = st.columns(2)
        with col1:
            option = st.selectbox(
                "Arama Nerede Yapılsın?",
                # ("Bütün Veriler", "Kanunlar", "Tebliğler", "Esaslar", "Yönetmelikler", "Yönergeler"),
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
                            # "Hepsi",
                            # "4734 Sayılı Kamu İhale Kanununun 62 nci Maddesinin (ı) Bendi Kapsamında Yapılacak Başvurulara İlişkin Tebliğ",
                            # "Doğrudan Temin Yöntemiyle Yapılacak Alımlara İlişkin Tebliğ",
                            # "Eşik Değerler ve Parasal Limitler Tebliği",
                            # "İhalelere Yönelik Başvurular Hakkında Tebliğ",
                            # "Kamu İhale Genel Tebliği",
                            # "Kamu Özel İş Birliği Projeleri ile Lisanslı İşler Kapsamında Gerçekleştirilen Yapım İşlerine İlişkin İş Deneyim Belgeleri Hakkında Tebliğ",
                            # "Yapım İşleri Benzer İş Grupları Tebliği"
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
  


                    if 'sidebar_content' not in st.session_state:
                        st.session_state.sidebar_content = 'İncelemek istediğiniz maddeyi seçiniz.'

                    response_txts = []
                    for idx, ret in enumerate(rets):
                        fileName = ret[0]
                        chunk = ret[1]
                        score = ret[2]
                        rank = idx + 1

                        # found_maddes = self.get_madde(chunk, fileName)
                        found_maddes = [
                            {"Title": "Madde", "Kanun No": "4734", "Kanun": "Madde içeriği"}, 
                            {"Title": "Madde2", "Kanun No": "47342", "Kanun": "Madde içeriği2"}
                            ]
                        

                        madde_txt = ""
                        option_txt = f"**Arama Kümesi:** {option}\n\n" if option2 is None else f"**Arama Kümesi:** {option} -> {option2}\n\n"

                        response_txt = f"Sonuçlar:\n\n__Sonuç-{rank}__\n\n{option_txt} **Bulunduğu Dosya:** {fileName.replace('.txt', '')}"
                        response_txt += f"\n\n**Madde:**\n\n"

                        if found_maddes:
                            for idy, madde in enumerate(found_maddes):
                                link_text = str(madde["Title"]) + " - Madde - "+ str(madde["Kanun No"]) +": " + str(madde["Kanun"])[:20]
                                sanitized_text = madde["Kanun"]
                                st.markdown("""
                                    <style>
                                    div.stButton > button:first-child {
                                        background: none;
                                        color: blue;
                                        border: none;
                                        padding: 0;
                                        font-size: 16px;
                                        text-decoration: underline;
                                        cursor: pointer;
                                    }
                                    </style>
                                """, unsafe_allow_html=True)

                                if st.button(link_text, key=f"button_{idx}_{idy}"):
                                    self.toggle_drawer()

                                # Display the drawer content if the drawer is open
                                if st.session_state['drawer_open']:
                                    with st.expander("", expanded=True):
                                        st.markdown(sanitized_text)
                                
                        else:
                            madde_txt = "Madde bulunamadı."

                        # madde = self.get_madde_via_embedding(chunk, fileName)

                        # response_txt = f"Sonuçlar:\n\nSonuç-{rank}:\n\nBulunduğu Dosya: {fileName.replace('.txt', '')}\n\n{chunk}"
                        



                        # response_txt += f"\n\n**Kısım:** \n\n{chunk}"

                        response_txts.append(response_txt)


                    # Render response
                    # response = st.write_stream(response)
                    # change respose_txt to generator
                    keywords = question.split()
                    # for response in response_txts:
                    #     keywords = question.split()
                    #     response = st.write(response)

                    for responset in response_txts:
                            highlighted_response = highlight_keywords(responset, keywords)
                            st.markdown(highlighted_response, unsafe_allow_html=True)  # Use markdown to allow HTML
                            st.session_state.messages.append(
                            {"role": "assistant", "content": responset})



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
