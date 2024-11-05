import ollama
import os
from glob import glob

# prompt = """
# You are a language model tasked with extracting specific sections from legal text. Your job is to identify and extract three pieces of information from the text provided:

# Madde Number: The number associated with each law article, appearing after the word "Madde".
# Title: The title of the section, which follows the Madde number or precedes the text description (usually bold or capitalized).
# Text: The body of the law, which is the description or clauses following the title.
# For each extracted item, format the output as follows:

# [Title], [Madde Number], [Text]

# Instructions:

# Extract the information for each "Madde" in the text.
# Do not include sub-clauses unless they are explicitly needed (i.e., ignore (a), (b) unless asked).
# Ignore irrelevant sections that do not follow the "Madde" structure.
# Example:

# İstisnalar, 3, a) Kanun kapsamına giren kuruluşlarca...

# Here’s the legal text to process:
# """

texta = """BİRİNCİ BÖLÜM

Amaç, Kapsam, Tanımlar ve İlkeler



	Amaç

	Madde 1- Bu Kanunun amacı, Kamu İhale Kanununa göre yapılan ihalelere ilişkin sözleşmelerin düzenlenmesi ve uygulanması ile ilgili esas ve usulleri belirlemektir.



	Kapsam

	Madde 2- Bu Kanun, Kamu İhale Kanununa tabi kurum ve kuruluşlar tarafından söz konusu Kanun hükümlerine göre yapılan ihaleler sonucunda düzenlenen sözleşmeleri kapsar.



	Tanımlar 

	Madde 3- Bu Kanunun uygulanmasında Kamu İhale Kanununda yer alan tanımlar geçerlidir.



	İlkeler

	Madde 4- Bu Kanuna göre düzenlenecek sözleşmelerde, ihale dokümanında yer alan şartlara aykırı hükümlere yer verilemez. 



	Bu Kanunda belirtilen haller dışında sözleşme hükümlerinde değişiklik yapılamaz ve ek sözleşme düzenlenemez. 



	Bu Kanun kapsamında yapılan kamu sözleşmelerinin tarafları, sözleşme hükümlerinin uygulanmasında eşit hak ve yükümlülüklere sahiptir.  İhale dokümanı ve sözleşme hükümlerinde bu prensibe aykırı maddelere yer verilemez.  Kanunun yorum ve uygulanmasında bu prensip göz önünde bulundurulur.



İKİNCİ BÖLÜM

Sözleşmelerin Düzenlenmesi



	Tip sözleşmeler 

	Madde 5- Bu Kanunun uygulanmasında uygulama birliğini sağlamak üzere mal veya hizmet alımları ile yapım işlerine ilişkin Tip Sözleşmeler Resmi Gazetede yayımlanır. 

		(Değişik ikinci fıkra: 20/11/2008-5812/31 md.) İdarelerce yapılacak sözleşmeler Tip Sözleşme hükümleri esas alınarak düzenlenir. Mal ve hizmet alımlarında, Kurumun uygun görüşü alınmak kaydıyla istekliler tarafından hazırlanması mutat olan sözleşmeler kullanılabilir.



	Sözleşme türleri

	Madde 6- Kamu İhale Kanununa göre yapılan ihaleler sonucunda; 

	a) Yapım işlerinde; uygulama projeleri ve bunlara ilişkin mahal listelerine dayalı olarak, işin tamamı için isteklinin teklif ettiği toplam bedel üzerinden anahtar teslimi götürü bedel sözleşme, 

	b) Mal veya hizmet alımı işlerinde, ayrıntılı özellikleri ve miktarı idarece belirlenen işin tamamı için isteklinin teklif ettiği toplam bedel üzerinden götürü bedel sözleşme,

c) Yapım işlerinde; ön veya kesin projelere ve bunlara ilişkin mahal listeleri ile birim fiyat tariflerine, mal veya hizmet alımı işlerinde ise işin ayrıntılı özelliklerine dayalı olarak; idarece hazırlanmış cetvelde yer alan her bir iş kaleminin miktarı ile bu iş kalemleri için istekli tarafından teklif edilen birim fiyatların çarpımı sonucu bulunan toplam bedel üzerinden birim fiyat sözleşme,

			d) (Ek: 1/6/2007-5680/3 md.; Değişik: 20/11/2008-5812/32 md.) Yapım işlerinde; niteliği itibarıyla iş kalemlerinin bir kısmı için anahtar teslimi götürü bedel, bir kısmı için birim fiyat teklifi alma yöntemleri birlikte uygulanmak suretiyle gerçekleştirilen ihaleler sonucunda karma sözleşme,

			e) (Ek: 20/11/2008-5812/32 md.) Çerçeve anlaşmaya dayalı olarak idare ile yüklenici arasında imzalanan münferit sözleşme,

Düzenlenir. 

	(Ek Fıkra: 1/6/2007-5680/3 md.) Çerçeve anlaşma ve münferit sözleşmede belirtilmesi zorunlu olan hususları belirlemeye Kurum yetkilidir. 

	Sözleşmede yer alması zorunlu hususlar

	Madde 7- Bu Kanuna göre düzenlenecek sözleşmelerde aşağıdaki hususların belirtilmesi zorunludur:

İşin adı, niteliği, türü ve miktarı, hizmetlerde iş tanımı.

İdarenin adı ve adresi.

	c) Yüklenicinin adı veya ticaret unvanı, tebligata esas adresi.

d) Varsa alt yüklenicilere ilişkin bilgiler ve sorumlulukları.

e) Sözleşmenin bedeli, türü ve süresi.

f) Ödeme yeri ve şartlarıyla avans verilip verilmeyeceği, verilecekse şartları ve miktarı.

g) Sözleşme konusu işler için ödenecekse fiyat farkının ne şekilde ödeneceği.

h) Ulaşım, sigorta, vergi, resim ve harç giderlerinden hangisinin sözleşme bedeline dahil olacağı.

i) Vergi, resim ve harçlar ile sözleşmeyle ilgili diğer giderlerin kimin tarafından ödeneceği.

j) Montaj, işletmeye alma, eğitim, bakım-onarım, yedek parça gibi destek hizmetlerine ait şartlar.

k) Kesin teminat miktarı ile kesin teminatın iadesine ait şartlar.

l) Garanti istenilen hallerde süresi ve garantiye ilişkin şartlar.

m) İşin yapılma yeri, teslim etme ve teslim alma şekil ve şartları.

n) Gecikme halinde alınacak cezalar.

o) (Değişik: 30/7/2003-4964/43 md.) Mücbir sebepler ve süre uzatımı verilebilme şartları, sözleşme kapsamında yaptırılacak iş artışları ile iş eksilişi durumunda karşılıklı yükümlülükler.

p) Denetim, muayene ve kabul işlemlerine ilişkin şartlar.

r) Yapım işlerinde iş ve işyerinin sigortalanması ile yapı denetimi ve sorumluluğuna ilişkin şartlar.

s) Sözleşmede değişiklik yapılma şartları.

t) Sözleşmenin feshine ilişkin şartlar.

u) Yüklenicinin sözleşme konusu iş ile ilgili çalıştıracağı personele ilişkin sorumlulukları.

v) İhale dokümanında yer alan bütün belgelerin sözleşmenin eki olduğu.

y) Anlaşmazlıkların çözümü.

z) (Ek: 23/04/2015--6645/33 md.) İş sağlığı ve güvenliğine ilişkin yükümlülükler.

"""

prompt = """ 
Sana verilen metindeki Madde'leri ayırt etmen ve çıkartman gerekiyor. Her bir Madde'nin başlığını, numarasını ve metnini çıkartman gerekiyor. Çıkarılan her bir öğe için çıktıyı aşağıdaki gibi biçimlendir:

[Başlık]|[Madde Numarası]|[Metin]

Örnek:

Bu bir örnek başlıktır.|Madde 3|Bu bir örnek metindir.

Orijinal metni hiçbir zaman değiştirme, yorumlamadan yaz.
Başka hiçbir şey söyleme, sadece cevabı yaz. İşte işlemek için verilen metin:
"""



files = glob("Data/output_txt/Kanun/**/*", recursive=True)
files = [file for file in files if os.path.isfile(file)]

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

    try:
        response = ollama.generate(
            model='nemotron', 
            prompt=(prompt + '\n\n' + text[:3000]),
            options = {
                'temperature': 0.0,
            })
    except:
         print("Aga... Hata var.")
    # print("Original text babushhhhhhh:" ,text)
    chat_output = response['response']
    # print(f"File: {file}")
    print("Response: ", chat_output)
    break