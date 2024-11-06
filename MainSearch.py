import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from txtai import Embeddings
import pandas as pd
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
import time
import ollama
from tqdm import tqdm
from glob import glob
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import time



class Config:
    # Retrive Types
    RetriveType_TOPK = 1
    RetriveType_THRESHOLD = 2
    RetriveType_TOPK_PLUS_THRESHOLD = 3
    RetriveType_SECTION_PLUS_THRESHOLD_TOPK = 4
    RetriveType_DOC = 5
    RetriveType_RERANK = 6

    # Options
    Option_All = "Bütün Veriler"
    Option_Kanun = "Kanunlar"
    Option_Teblig = "Tebliğler"
    Option_Esas = "Esaslar"
    Option_Yonetmelik = "Yönetmelikler"
    Option_Yonerge = "Yönergeler"

    # Kanun Options
    Option_Kanun_Hepsi = "Hepsi"
    Option_Kanun_4734 = "4734"
    Option_Kanun_4735 = "4735"

    # Teblig Options
    Option_Teblig_Hepsi = "Hepsi"
    Option_Teblig_4734 = "4734 Sayılı Kamu İhale Kanununun 62 nci Maddesinin (ı) Bendi Kapsamında Yapılacak Başvurulara İlişkin Tebliğ"
    Option_Teblig_Dogrudan = "Doğrudan Temin Yöntemiyle Yapılacak Alımlara İlişkin Tebliğ"
    Option_Teblig_Esik = "Eşik Değerler ve Parasal Limitler Tebliği"
    Option_Teblig_Ihalelere = "İhalelere Yönelik Başvurular Hakkında Tebliğ"
    Option_Teblig_KamuIhale = "Kamu İhale Genel Tebliği"
    Option_Teblig_KamuOzel = "Kamu Özel İş Birliği Projeleri ile Lisanslı İşler Kapsamında Gerçekleştirilen Yapım İşlerine İlişkin İş Deneyim Belgeleri Hakkında Tebliğ"
    Option_Teblig_Yapim = "Yapım İşleri Benzer İş Grupları Tebliği"

    # Esas Options
    Option_Esas_Hepsi = "Hepsi"

    # Yonetmelik Options
    Option_Yonetmelik_Hepsi = "Hepsi"
    Option_Yonetmelik_Ihale = "İhale Uygulama Yönetmelikleri"
    Option_Yonetmelik_Muayene = "Muayene ve Kabul Yönetmelikleri"
    Option_Yonetmelik_Ihalelere = "İhalelere Yönelik Başvurular Hakkında Yönetmelik"

    # Yonerge Options
    Option_Yonerge_Hepsi = "Hepsi"
    Option_Yonerge_Itiraz = "İtirazen Şikayet Başvuru Bedelinin İadesine İlişkin Yönerge"
    Option_Yonerge_Yurt = "Yurt Dışında Yapım İşlerinden Elde Edilen İş Deneyim Belgelerinin Belgelerin Sunuluş Şekline Uygunluğunu Tevsik Amacıyla EKAP'a Kaydedilmesine İlişkin Yönerge"

    

    # Model Names
    ModelName_BAAI_BGE_M3 = 'BAAI/bge-m3'

    # Index File
    IndexFile_BAAI_BGE_M3 = 'Data/Indexes/'

    # Chunk Types
    ChunkType_512 = 1
    ChunkType_WHOLE = 2

    # Chunking Methods
    ChunkingMethod_OLD = "Eski"
    ChunkingMethod_NEW = "Yeni"

    def calculate_runtime(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time
            print(f"Function '{func.__name__}' executed in: {runtime:.4f} seconds")
            return result
        return wrapper


class Chunking():
    def __init__(self, chunk_type, txt_folder="/media/alperk/Disk/KiK/KiK_Application/Data/converted_kik_data_txt_format"):
        self.chunk_type = chunk_type
        self.txt_folder = glob(txt_folder + '/*.txt')

    def chunk(self, overlap=2):
        if self.chunk_type == Config.ChunkType_512:
            df = self.chunk_512(overlap)
        elif self.chunk_type == Config.ChunkType_WHOLE:
            df = self.chunk_whole()

        return df

    def chunk_512(self, overlap):
        df = pd.DataFrame(columns=['FileName', 'Chunk'])
        for txt_file in self.txt_folder:
            with open(txt_file, 'r') as file:
                text = file.read()
                chunks = self.split_text(text, 512, overlap)
                for chunk in chunks:
                    df = df.append({'FileName': txt_file.split('/')[-1], 'Chunk': chunk}, ignore_index=True)

    def chunk_whole(self):
        df = pd.DataFrame(columns=['FileName', 'Chunk'])
        for txt_file in self.txt_folder:
            with open(txt_file, 'r') as file:
                text = file.read()
                df = df.append({'FileName': txt_file.split('/')[-1], 'Chunk': text}, ignore_index=True)

    def split_text(self, text, max_len, overlap):
        text = self.preprocess(text)
        words = text.split()
        chunks = []
        chunk = ""
        start_idx = 0 
        
        while start_idx < len(words):
            chunk = ""
            idx = start_idx
            while idx < len(words) and len(chunk) + len(words[idx]) <= max_len:
                chunk += words[idx] + " "
                idx += 1
            
            chunks.append(chunk.strip()) 
            start_idx = max(0, idx - overlap)
        return chunks


    def preprocess(self, text: str) -> str:
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[_\[\]\{\}]", " ", text)
        text = re.sub(r"\S*\.{2,}", "", text)
        text = re.sub(r"\.{2,}\s*", " ", text)
        text = re.sub(r"\…", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

class MainSearch():
    def __init__(self, model_name, chunk_type, chunking_method):
        self.model_name = model_name
        self.chunker_df = None    
        self.chunking_method = chunking_method

        print(f"Model creating for {model_name}")
        self.embedding = Embeddings(path=model_name, hybrid=True, content=True, trust_remote_code=True)

        if chunking_method == Config.ChunkingMethod_OLD:
            self.df_file_name = pd.read_csv("Data/kik_512_chunk_17_09_24.csv", index_col=False)
        elif chunking_method == Config.ChunkingMethod_NEW:
            self.df_file_name = pd.read_csv("Data/new_version/chunked_kik_all_madde_filtered.csv", index_col=False)
        else:
            print("Invalid Chunking Method")
            return
        
        self.IndexBasePath = "Data/Indexes/"


        # self.chunker = Chunking(chunk_type)
        # index_path = self.index(model_name, indexed_csv_file, chunk_type)
        # print(f"Index loade.txtd from {index_path}")


        # Initialize the Hugging Face reranker model
        reranker_model_name = "BAAI/bge-reranker-v2-m3"
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Is cuda available?: ",torch.cuda.is_available())
        self.reranker_model.to(self.device)

    def index(self, model_name, csv_file, chunk_type):
        self.chunker_df = self.chunker.chunk()
        df = self.chunker_df.copy()
        df.drop(df.columns[0], axis=1, inplace=True)
        df.rename(columns={df.columns[0]: 'text'}, inplace=True)
        ds = df["text"].apply(lambda x: {"text": x} if isinstance(x, str) else None).dropna().tolist()

        self.embedding.index(tqdm(ds, total=len(ds)))
        self.embedding.save(f"Index/{model_name.split('/')[-1]}")
        return f"Index/{model_name.split('/')[-1]}"
    
    def prepare_rerank_input(self, query, document):
        return [query,document]
            
    def run(self, test_csv, retrive_type = Config.RetriveType_TOPK, top_k = 5, num_chunk=100, hybrit_rate=0.5, chunk_score_threshold = 0.4, exp_name = "exp", verbose=False, slow_run=False, chat=False):
        print(f"Running test for {self.model_name}, with {num_chunk} chunks, weights: {hybrit_rate}, threshold: {chunk_score_threshold}")
        total_hit_rate = 0
        total_mrr_rate = 0
        counter = 0 

        exp_name = exp_name + "_" + datetime.now().strftime("%d-%m-%y")
        file_name = f"{exp_name}_{hybrit_rate}_{self.model_name.split('/')[-1]}_aeb.csv"
        test_df = pd.read_csv(test_csv)

        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header_row = ['query', 'passage']
            for i in range(1, 6):
                header_row.append(f'result{i}')
            header_row += ['matched_index', 'mrr', 'hit']
            writer.writerow(header_row)
            print("-"*10, end='\n\n')

            indexed_dict = {}
            
            for row in test_df.iterrows():
                counter += 1
                matched_index = 0

                if not verbose:
                    print(f"Processing {counter}/{len(test_df)}", end='\r')
                
                test_file_name = row[1]['FileName']
                query = row[1]['Question']
                context = row[1]['Chunk']

                results = self.embedding.search(query, weights=hybrit_rate, limit=num_chunk)
                # if retrive_type != Config.RetreiveType_DOC:
                #     results = self.embedding.search(query, weights=hybrit_rate, limit=num_chunk)
                # else:
                #     results = []
                #     doc_chunk_dict = {}
                #     if self.chunker_df:
                #         for idx, chunk in self.chunker_df.iterrows():
                #             if chunk['FileName'] not in doc_chunk_dict:
                #                 doc_chunk_dict[chunk['FileName']] = []
                #             doc_chunk_dict[chunk['FileName']].append(chunk['Chunk'])
                    
                #     for doc_name, chunks in doc_chunk_dict.items():
                #         doc_text = ' '.join(chunks)
                #         avg_similarity = 0
                #         for chunk in chunks:
                #             similarity = self.embedding.similarity(query, chunk)
                #             avg_similarity += similarity

                #         avg_similarity /= len(chunks)
                #         results.append({'text': doc_text, 'score': avg_similarity, 'FileName': doc_name})



                if retrive_type == Config.RetriveType_TOPK:
                    chat_text_arr = [r['text'] for r in results]
                    chat_text_txt = '\n\n'.join(chat_text_arr)
                    chat_text_txt += '\n\n' + query
                    current_top_k = 0
                    for idx, result in enumerate(results, start=1):
                        if current_top_k >= top_k:
                            break
                        file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                        predicted_file_name = file_name_row['FileName'].values[0]
                        if result['text'] == context:
                            if verbose:
                                print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])

                            matched_index = idx
                            total_hit_rate += 1
                            total_mrr_rate += 1 / idx 
                            break
                        current_top_k += 1

                elif retrive_type == Config.RetriveType_THRESHOLD:
                    for idx, result in enumerate(results, start=1):
                        if result['score'] >= chunk_score_threshold:
                            file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                            predicted_file_name = file_name_row['FileName'].values[0]
                            if result['text'] == context:
                                if verbose:
                                    print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])
                                if first_file_name == "":
                                    first_file_name = predicted_file_name

                                matched_index = idx
                                total_hit_rate += 1
                                total_mrr_rate += 1 / idx 
                                break

                elif retrive_type == Config.RetriveType_TOPK_PLUS_THRESHOLD:
                    current_top_k = 0
                    for idx, result in enumerate(results, start=1):
                        if current_top_k >= top_k:
                            break
                        if result['score'] >= chunk_score_threshold:
                            file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                            predicted_file_name = file_name_row['FileName'].values[0]
                            if result['text'] == context:
                                if verbose:
                                    print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])
                                if first_file_name == "":
                                    first_file_name = predicted_file_name

                                matched_index = idx
                                total_hit_rate += 1
                                total_mrr_rate += 1 / idx 
                                break
                        current_top_k += 1

                elif retrive_type == Config.RetriveType_SECTION_PLUS_THRESHOLD_TOPK:
                    first_file_name = ""
                    thresholed_results = [r for r in results if r['score'] >= chunk_score_threshold]
                    thresholed_results = [dict(r, **{'file_name': self.df_file_name.loc[self.df_file_name['Chunk'] == r['text']]['FileName'].values[0]}) for r in thresholed_results]
                    unique_file_names = set([r['file_name'] for r in thresholed_results])
                    current_top_k = 0
                    for s_file_name in unique_file_names:
                        if current_top_k >= top_k:
                            break
                        for idx, result in enumerate(thresholed_results, start=1):
                            if result['file_name'] == s_file_name:
                                if result['text'] == context:
                                    if verbose:
                                        print(result['id'], ' - ', result['score'], ' - ', result['file_name'], ' - ', result['text'][:25])
                                    if first_file_name == "":
                                        first_file_name = s_file_name

                                    matched_index = idx
                                    total_hit_rate += 1
                                    total_mrr_rate += 1 / idx 
                                    break
                                current_top_k += 1
                                break
                    
                # elif retrive_type == Config.RetriveType_DOC:
                #     current_top_k = 0
                #     for idx, result in enumerate(results, start=1):
                #         if result['FileName'] == test_file_name:
                #             if result['text'] == context:
                #                 matched_index = idx
                #                 total_hit_rate += 1
                #                 total_mrr_rate += 1 / idx 
                #                 break
                #             current_top_k += 1              

                elif retrive_type == Config.RetriveType_RERANK:
                    current_top_k = 0

                    chat_text_arr = [r['text'] for r in results]
                    chat_text_txt = '\n\n'.join(chat_text_arr)
                    chat_text_txt += '\n\n' + query

                    rerank_inputs = [self.prepare_rerank_input(query, result["text"]) for result in results]

                    with torch.no_grad():
                        inputs = self.tokenizer(rerank_inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)
                        inputs = {key: value.to(self.device) for key, value in inputs.items()}
                        scores = self.reranker_model(**inputs).logits.view(-1).float()
                        results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
                       
                    
                    results = [res[1] for res in results] 

    
                    for idx, result in enumerate(results, start=1):
                        file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                        predicted_file_name = file_name_row['FileName'].values[0]
                        result['FileName'] = predicted_file_name

   

                    for idx, result in enumerate(results, start=1):                        

                        if current_top_k >= top_k:
                            break
                        file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                        predicted_file_name = file_name_row['FileName'].values[0]
                        if result['text'] == context:
                            if verbose:
                                print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])

                            matched_index = idx
                            total_hit_rate += 1
                            total_mrr_rate += 1 / idx 
                            break
                        current_top_k += 1
                    

                    


                indexed_dict[str(matched_index)] = indexed_dict.get(str(matched_index), 0) + 1

                if verbose:
                    print(f"MRR: {1 / (matched_index+1e-9)}")
                    print(f"Hit: {1 if matched_index is not None else 0}", end='\n\n')      
                if slow_run:
                    time.sleep(1)
                if chat:
                    response = ollama.chat(model='main-rag', messages=[
                    {
                        'role': 'user',
                        'content': chat_text_txt
                    },
                    ])

                    print(f"Query: {query}")
                    chat_output = response['message']['content']
                    print("Response: ", chat_output)
                    input("Press Enter to continue...")

            average_hit_rate = total_hit_rate / counter
            average_mrr_rate = total_mrr_rate / counter
            print(f"\nAverage Hit Rate: {average_hit_rate}")
            print(f"Average MRR Rate: {average_mrr_rate}", end='\n\n')
            
            # sort indexed_dict by key
            indexed_dict = dict(sorted(indexed_dict.items(), key=lambda item: int(item[0])))

            for key, value in indexed_dict.items():
                if key != "0":
                    print(f"{key} - Count: {value} - Percentage: {round((value / (counter - indexed_dict['0'])) * 100, 2)}%")


            writer.writerow([query, context, results[0]['text'], results[1]['text'], results[2]['text'], results[3]['text'], results[4]['text'], matched_index, (1 / (matched_index+1e-9)), 1 if matched_index is not None else 0])
    
        print(f"Results saved to {file_name}")

    
    @Config.calculate_runtime
    def embedding_search(self, query, weights, limit):
        return self.embedding.search(query, weights=weights, limit=limit)
    
    @Config.calculate_runtime
    def embedding_search_new(self, query, weights, limit, options):
        temp_res =  self.embedding.search(query, weights=weights, limit=limit)
        if options:
            re_res = []
            for res in temp_res:
                print("")
        else:
            return temp_res
    

    @Config.calculate_runtime
    def retreive(self, query, option=Config.Option_All, option2=None, option3=[], retrive_type = Config.RetriveType_TOPK, top_k = 5, num_chunk=100, hybrit_rate=0.5, temperature=0.7, chunk_score_threshold = 0.4, exp_name = "exp", verbose=False, slow_run=False, chat=False):
        self.option3 = option3
        # self.embedding.load(path='/data/Workspace/aebayar/KiK_RaG/Chat_GPT_Indexes/bKik_index_model_bge-m3_01-10-2024', repo_type="local")
        if self.chunking_method == Config.ChunkingMethod_OLD:
            if option == Config.Option_All: 
                self.embedding.load(path=self.IndexBasePath+'All', repo_type="local")
            elif option == Config.Option_Kanun:
                if option2 == Config.Option_Kanun_Hepsi:
                    self.embedding.load(path=self.IndexBasePath+'Kanunlar', repo_type="local")
                elif option2 == Config.Option_Kanun_4734:
                    self.embedding.load(path=self.IndexBasePath+'Kanunlar_4734', repo_type="local")
                elif option2 == Config.Option_Kanun_4735:
                    self.embedding.load(path=self.IndexBasePath+'Kanunlar_4735', repo_type="local")
            elif option == Config.Option_Teblig:
                if option2 == Config.Option_Teblig_Hepsi:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler', repo_type="local")
                elif option2 == Config.Option_Teblig_4734:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_4734', repo_type="local")
                elif option2 == Config.Option_Teblig_Dogrudan:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_Dogrudan', repo_type="local")
                elif option2 == Config.Option_Teblig_Esik:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_Esik', repo_type="local")
                elif option2 == Config.Option_Teblig_Ihalelere:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_Ihalelere', repo_type="local")
                elif option2 == Config.Option_Teblig_KamuIhale:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_KamuIhale', repo_type="local")
                elif option2 == Config.Option_Teblig_KamuOzel:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_KamuOzel', repo_type="local")
                elif option2 == Config.Option_Teblig_Yapim:
                    self.embedding.load(path=self.IndexBasePath+'Tebligler_Yapim', repo_type="local")
            elif option == Config.Option_Esas:
                if option2 == Config.Option_Esas_Hepsi:
                    self.embedding.load(path=self.IndexBasePath+'Esaslar', repo_type="local")
            elif option == Config.Option_Yonetmelik:
                if option2 == Config.Option_Yonetmelik_Hepsi:
                    self.embedding.load(path=self.IndexBasePath+'Yonetmelikler', repo_type="local")
                elif option2 == Config.Option_Yonetmelik_Ihale:
                    self.embedding.load(path=self.IndexBasePath+'Yonetmelikler_Ihale', repo_type="local")
                elif option2 == Config.Option_Yonetmelik_Muayene:
                    self.embedding.load(path=self.IndexBasePath+'Yonetmelikler_Muayene', repo_type="local")
                elif option2 == Config.Option_Yonetmelik_Ihalelere:
                    self.embedding.load(path=self.IndexBasePath+'Yonetmelikler_Ihalelere', repo_type="local")
            elif option == Config.Option_Yonerge:
                if option2 == Config.Option_Yonerge_Hepsi:
                    self.embedding.load(path=self.IndexBasePath+'Yonergeler', repo_type="local")
                elif option2 == Config.Option_Yonerge_Itiraz:
                    self.embedding.load(path=self.IndexBasePath+'Yonergeler_Itiraz', repo_type="local")
                elif option2 == Config.Option_Yonerge_Yurt:
                    self.embedding.load(path=self.IndexBasePath+'Yonergeler_Yurt', repo_type="local")
        elif self.chunking_method == Config.ChunkingMethod_NEW:
            self.embedding.load(path='Data/new_version/new_madde_logic_bge-m3_06-11-2024', repo_type="local")
        else:
            print("Invalid Chunking Method")
            return

        
        print(f"Running test for {self.model_name}, with {num_chunk} chunks, weights: {hybrit_rate}, threshold: {chunk_score_threshold}")
        total_hit_rate = 0
        total_mrr_rate = 0
        counter = 0 
        
        return_value = []

        if self.chunking_method == Config.ChunkingMethod_OLD:
            results = self.embedding_search(query, weights=hybrit_rate, limit=num_chunk)
        elif self.chunking_method == Config.ChunkingMethod_NEW:
            results = self.embedding_search_new(query, hybrit_rate, num_chunk, self.option3)
        else:
            print("Invalid Chunking Method")
            return
        
        # results = self.embedding.search(query, weights=hybrit_rate, limit=num_chunk)
        print("LOGAEB: res len: ", len(results))

        if retrive_type == Config.RetriveType_TOPK:
            chat_text_arr = [r['text'] for r in results]
            chat_text_txt = '\n\n'.join(chat_text_arr)
            chat_text_txt += '\n\n' + query
            current_top_k = 0

            for idx, result in enumerate(results, start=1):
                if current_top_k >= top_k:
                    break
                file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                predicted_file_name = file_name_row['FileName'].values[0]
                current_top_k += 1
                return_value.append((predicted_file_name, result['text'], result['score']))

        elif retrive_type == Config.RetriveType_THRESHOLD:
            for idx, result in enumerate(results, start=1):
                if result['score'] >= chunk_score_threshold:
                    
                    file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                    rows_index = self.df_file_name[self.df_file_name['Chunk'] == result['text']].index
        
                    predicted_file_name = file_name_row['FileName'].values[0]
                    if verbose:
                        print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])
                    if first_file_name == "":
                        first_file_name = predicted_file_name

                    matched_index = idx
                    total_hit_rate += 1
                    total_mrr_rate += 1 / idx 
                    return_value.append((predicted_file_name, result['text'], result['score']))

        elif retrive_type == Config.RetriveType_TOPK_PLUS_THRESHOLD:
            current_top_k = 0
            for idx, result in enumerate(results, start=1):
                if current_top_k >= top_k:
                    break
                if result['score'] >= chunk_score_threshold:
                    file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                    predicted_file_name = file_name_row['FileName'].values[0]
                   
                    if verbose:
                        print(result['id'], ' - ', result['score'], ' - ', predicted_file_name, ' - ', result['text'][:25])
                    if first_file_name == "":
                        first_file_name = predicted_file_name

                    matched_index = idx
                    total_hit_rate += 1
                    total_mrr_rate += 1 / idx 
                    return_value.append((predicted_file_name, result['text'], result['score']))
                current_top_k += 1

        elif retrive_type == Config.RetriveType_SECTION_PLUS_THRESHOLD_TOPK:
            first_file_name = ""
            chat_text_txt = ""

            thresholed_results = [r for r in results if r['score'] >= chunk_score_threshold]
            # add file_name to the thresholded results
            thresholed_results = [dict(r, **{'file_name': self.df_file_name.loc[self.df_file_name['Chunk'] == r['text']]['FileName'].values[0]}) for r in thresholed_results]
            print("thresholed_results: ", len(thresholed_results))
            unique_file_names = set([r['file_name'] for r in thresholed_results])
            current_top_k = 0
            print("unique_file_names: ", len(unique_file_names))
            for s_file_name in unique_file_names:
                if current_top_k >= top_k:
                    break
                print("thresholded_results: ", len(thresholed_results))
                for idx, result in enumerate(thresholed_results, start=1):
                    if result['file_name'] == s_file_name:
                        if verbose:
                            print(result['id'], ' - ', result['score'], ' - ', result['file_name'], ' - ', result['text'][:25])
                        if first_file_name == "":
                            first_file_name = s_file_name

                        matched_index = idx
                        total_hit_rate += 1
                        total_mrr_rate += 1 / idx 
                    current_top_k += 1
                    return_value.append((s_file_name, result['text'], result['score']))
                    chat_text_txt += ('\n\n' + result['text'])
                
        elif retrive_type == Config.RetriveType_RERANK:
            current_top_k = 0
            chat_text_txt = ""
            rerank_inputs = [self.prepare_rerank_input(query, result["text"]) for result in results]

            with torch.no_grad():
                inputs = self.tokenizer(rerank_inputs, padding=True, truncation=True, return_tensors="pt", max_length=512)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                scores = self.reranker_model(**inputs).logits.view(-1).float()
                results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
                
            
            results = [res[1] for res in results] 


            for idx, result in enumerate(results, start=1):
                file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                try:
                    predicted_file_name = file_name_row['FileName'].values[0]
                except:
                    predicted_file_name = "Unknown"
                result['FileName'] = predicted_file_name


            for idx, result in enumerate(results, start=1):                        
                if current_top_k >= top_k:
                    break
                file_name_row = self.df_file_name.loc[self.df_file_name['Chunk'] == result['text']]
                try:
                    predicted_file_name = file_name_row['FileName'].values[0]
                except:
                    predicted_file_name = "Unknown"
                current_top_k += 1
                return_value.append((predicted_file_name, result['text'], result['score']))

                chat_text_txt += ('\n\n' + result['text'])
        
        chat_text_txt += '\n\n' + query

        if chat:
            response = ollama.generate(
                model='nemotron', 
                prompt=chat_text_txt,
                options = {
                    'temperature': temperature,
                })

            chat_output = response['response']

            

            print(f"Query: {query}")
            print("Response: ", chat_output)
            file_names = [r[0] for r in return_value]

            chat_output += ('\n\n Model Input: \n\n')
            
            for idx, fn in enumerate(file_names):
                chat_output += '\n\n'
                chat_output += '*'*150
                chat_output += '\n\n'
                chat_output += f"[{idx+1}] - Dosya Adı: {fn}\n\n"
                chat_output += return_value[idx][1]
                chat_output += '\n\n'

            
            chat_output += ('\n\n Direct Input: \n\n')
            chat_output += chat_text_txt    

            return chat_output, file_names

        return return_value