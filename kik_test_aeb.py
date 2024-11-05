from MainSearch import MainSearch, Config
import pandas as pd


"""
Config:
Retrive Types:
    - TOPK
    - THRESHOLD
    - TOPK_PLUS_THRESHOLD
    - SECTION_PLUS_THRESHOLD_TOPK
Model Names:
    - BAAI_BGE_M3
"""

index_file = '/data/Workspace/aebayar/KiK_RaG/Chat_GPT_Indexes/bKik_index_model_bge-m3_01-10-2024'
test_dataset ='Data/kik_main_w_chatgpt_test.csv'
csv_file = 'Data/kik_512_chunk_17_09_24.csv'

model_name = Config.ModelName_BAAI_BGE_M3
main_search = MainSearch(model_name=model_name, indexed_csv_file=csv_file, chunk_type=Config.ChunkType_512)

# main_search.run(test_dataset, 
#                 retrive_type=Config.RetriveType_TOPK,
#                 num_chunk=5,
#                 top_k=5,
#                 hybrit_rate=0.5, 
#                 chunk_score_threshold=0.6, 
#                 exp_name="MAIN_SEARCH",
#                 slow_run=False, 
#                 verbose=False,
#                 chat=False)

question = "İhale sürecinde, ihale dokümanlarının hazırlanması ve ihale sürecinin yürütülmesi aşamalarında, ihale komisyonunun görev ve yetkileri nelerdir?"

rets = main_search.retreive(query=question,
            option=Config.Option_All,
            option2=None,
            retrive_type=Config.RetriveType_SECTION_PLUS_THRESHOLD_TOPK,
            top_k=5, 
            num_chunk=50, 
            hybrit_rate=0.5, 
            chunk_score_threshold=0.4, 
            exp_name="exp", 
            verbose=False, 
            slow_run=False, 
            chat=True)

print("\n"*10)
print("Results:", rets)


# csv_file_path = "/data/Workspace/aebayar/KiK_RaG/21_questions_from_kik.csv"
# df = pd.read_csv(csv_file_path)
# questions = df['Question'].tolist()

# save_csv = "/data/Workspace/aebayar/KiK_RaG/21_questions_from_kik_results.csv"
# save_df = pd.DataFrame(columns=['Soru', 'Sonuç Sırası', 'Dosya İsmi', 'Skor', 'Chunk'])

# for question in questions:
#     rets = main_search.retreive(question,
#                                 retrive_type=Config.RetriveType_TOPK, 
#                                 top_k=5, 
#                                 num_chunk=5, 
#                                 hybrit_rate=0.5, 
#                                 chunk_score_threshold=0.4, 
#                                 exp_name="exp", 
#                                 verbose=False, 
#                                 slow_run=False, 
#                                 chat=False)
    
#     for idx, ret in enumerate(rets):
#         fileName = ret[0]
#         chunk = ret[1]
#         score = ret[2]
#         rank = idx + 1

#         new_row = pd.DataFrame({'Soru': [question], 'Sonuç Sırası': [rank], 'Dosya İsmi': [fileName], 'Skor': [round(score, 2)], 'Chunk': [chunk]})
#         save_df = pd.concat([save_df, new_row], ignore_index=True)

#         print(f"Question: {question}") 
#         print("Retrive Results:")
#         print(f"File Name: {fileName}")
#         print(f"Score: {score}")
#         print(f"Chunk: {chunk}")
    
#     # Adding an empty row
#     empty_row = pd.DataFrame({'Soru': [''], 'Sonuç Sırası': [''], 'Dosya İsmi': [''], 'Skor': [''], 'Chunk': ['']})
#     save_df = pd.concat([save_df, empty_row], ignore_index=True)

# save_df.to_csv(save_csv, index=False)