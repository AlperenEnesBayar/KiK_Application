"""
Runs a RAG application backed by a txtai Embeddings database.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import platform
import re

from glob import glob
from uuid import UUID
import streamlit as st

from MainSearch import MainSearch, Config

from download_button import download_button

# Build logger
logger = st.logger.get_logger(__name__)

def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, f"<span style='background-color: yellow'>{keyword}</span>")
        # text = re.sub(rf'\b{keyword}\b', f"<span style='background-color: yellow'>{keyword}</span>", text)
    return text

class AutoId:
    @staticmethod
    def valid(uid):
        try:
            return UUID(str(uid))
        except ValueError:
            pass
        return isinstance(uid, int) or uid.isdigit()


class Application:
    def __init__(self):
        """
        Creates a new application.
        """
        self.index_file = '/data/Workspace/aebayar/KiK_RaG/Chat_GPT_Indexes/bKik_index_model_bge-m3_01-10-2024'
        model_name = Config.ModelName_BAAI_BGE_M3
        csv_file = '/data/Workspace/aebayar/KiK_RaG/Data/zup.csv' # 'Data/kik_512_chunk_17_09_24.csv'

        self.context = int(os.environ.get("CONTEXT", 10))
        self.main_search = MainSearch(model_name=model_name, indexed_csv_file=csv_file, chunk_type=Config.ChunkType_512)

        self.all_file_paths = glob("Data/Akıllı Arama Mevzuat/**/*", recursive=True)
        self.search_type = Config.RetriveType_RERANK
        self.num_chunks = 50
        self.top_k = 5
        self.hybrit_rate = 0.5
        self.temp = 0.7

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


    def run(self):
        st.divider()
        pr_col1, pr_col2, pr_col3 = st.columns([0.2, 0.4, 0.4], vertical_alignment="center")
        with pr_col1:
            radio_option = st.radio(
                "Select an option:",
                ('Reranker', 'Section')
            )
            if radio_option == 'Reranker':
                self.search_type = Config.RetriveType_RERANK
            else:
                self.search_type = Config.RetriveType_SECTION_PLUS_THRESHOLD_TOPK
        
        with pr_col2:
            self.num_chunks = st.number_input('Enter the Number of Chunks:', min_value=1, value=self.num_chunks)
            self.top_k = st.number_input('Enter the Top K:', min_value=1, value=self.top_k)

        with pr_col3:
            self.hybrit_rate = st.slider('Select the hybrid rate:', min_value=0.01, max_value=1.0, value=self.hybrit_rate, step=0.01)
            self.temp = st.slider('Select the temperature rate:', min_value=0.01, max_value=1.0, value=self.temp, step=0.01)
        
        st.divider()

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
                        Config.Option_Esas_Hepsi
                        ))
            elif option == "Yönetmelikler":
                with col2:
                    option2 = st.selectbox("Yönetmelikler", (
                        Config.Option_Yonetmelik_Hepsi,
                        Config.Option_Yonetmelik_Ihale,
                        Config.Option_Yonetmelik_Muayene,
                        Config.Option_Yonetmelik_Ihalelere
                        ))
            elif option == "Yönergeler":
                with col2:
                    option2 = st.selectbox("Yönergeler", (
                        Config.Option_Yonerge_Hepsi,
                        Config.Option_Yonerge_Itiraz,
                        Config.Option_Yonerge_Yurt
                        ))
            st.write(f'Arama Kümesi {option} -> {option2} olarak seçildi.')
        else:
            st.write(f'Arama Kümesi {option} olarak seçildi.')

        st.divider()
   

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", 
                 "content": (f"Kamu İhale Kurumu Akıllı Arama Motoru 🚀'na soru sormaya başlayabilirsiniz.")
                }]

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

            
                # Check for Graph RAG
                print("LOGAEB: ", question)

                placeholder = st.empty()        
                placeholder.markdown("Aranıyor... Düşünüyorum...")            
                rets, file_names = self.main_search.retreive(query=question,
                            option=option,
                            option2=option2,
                            retrive_type=self.search_type,
                            top_k=self.top_k, 
                            num_chunk=self.num_chunks, 
                            hybrit_rate=self.hybrit_rate, 
                            chunk_score_threshold=0.4, 
                            temperature=self.temp,
                            exp_name="exp", 
                            verbose=False, 
                            slow_run=False, 
                            chat=True)
                
                placeholder.markdown(rets, unsafe_allow_html=True)
                
                files =  []
                found_file_names = []
                for fn in file_names:
                    # find path from file name
                    file_path = None
                    for file_path in self.all_file_paths:
                        if fn in file_path:
                            break
                    if file_path:
                        if fn not in found_file_names:
                            files.append(open(file_path, "rb"))
                            found_file_names.append(fn)

                st.markdown(f"**Toplam {len(files)} dosya bulundu.**")
                for idx, (file, fn) in enumerate(zip(files, found_file_names)):
                    download_button_str = download_button(file.read(), fn.replace('.txt', ''), "["+ str(idx+1)+"]"+" - Dosya Adı: " + fn.replace('.txt', ''))
                    st.markdown(download_button_str, unsafe_allow_html=True)


                st.session_state.messages.append(
                    {"role": "assistant", "content": rets}
                    )


@st.cache_resource(show_spinner="Modeller yükleniyor... Veri setleri oluşturuluyor...")
def create():
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
