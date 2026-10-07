import re
from collections import Counter

import gensim
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import trafilatura
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

st.set_page_config(page_title="Klasifikasi Berita", layout="wide")

st.header("Klasifikasi Berita")

CONTOH = (
    "Timnas Indonesia berhasil mengalahkan lawannya dalam pertandingan sepak bola "
    "semalam. Pelatih memuji performa para pemain yang tampil solid sepanjang "
    "pertandingan di stadion."
)


@st.cache_resource
def load_models():
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    clf_tfidf = joblib.load("models/naive_bayes_tfidf.pkl")
    w2v = gensim.models.Word2Vec.load("models/skipgram.model")
    clf_w2v = joblib.load("models/naive_bayes_skipgram.pkl")
    stopwords = set(StopWordRemoverFactory().get_stop_words())
    stemmer = StemmerFactory().create_stemmer()
    return tfidf, clf_tfidf, w2v, clf_w2v, stopwords, stemmer


tfidf, clf_tfidf, w2v, clf_w2v, stopwords, stemmer = load_models()


# ---------------------------------------------------------------- pipeline
def run_pipeline(raw):
    """Jalankan preprocessing + prediksi, simpan hasil tiap tahap."""
    r = {"raw": raw}
    r["lower"] = str(raw).lower()
    r["no_url"] = re.sub(r"http\S+|www\.\S+", " ", r["lower"])
    r["clean"] = re.sub(r"[^a-z\s]", " ", r["no_url"])

    words = r["clean"].split()
    r["words"] = words
    r["removed"] = [w for w in words if w in stopwords or len(w) <= 2]
    kept = [w for w in words if w not in stopwords and len(w) > 2]
    r["kept"] = kept

    cache = {}
    for w in set(kept):
        cache[w] = stemmer.stem(w)
    r["stem_map"] = cache
    r["tokens"] = [cache[w] for w in kept]

    joined = " ".join(r["tokens"])
    r["vec_tfidf"] = tfidf.transform([joined]).toarray()

    in_vocab = [w for w in r["tokens"] if w in w2v.wv]
    r["in_vocab"], r["oov"] = in_vocab, [w for w in r["tokens"] if w not in w2v.wv]
    r["vec_w2v"] = (
        np.mean([w2v.wv[w] for w in in_vocab], axis=0)
        if in_vocab
        else np.zeros(w2v.vector_size)
    ).reshape(1, -1)

    for key, clf, vec in (
        ("tfidf", clf_tfidf, r["vec_tfidf"]),
        ("w2v", clf_w2v, r["vec_w2v"]),
    ):
        r[f"pred_{key}"] = clf.predict(vec)[0]
    return r


def extract_url(url):
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    doc = trafilatura.bare_extraction(resp.text, with_metadata=True)
    if not doc:
        return None, None

    def get(key):
        return doc.get(key) if isinstance(doc, dict) else getattr(doc, key, None)

    text = get("text")
    if not text:
        return None, None
    text = re.sub(
        r"Scroll atau gunakan tombol\s*\[\]\s*\[\]\s*serta klik panah di sisi kanan\s*untuk menjelajahi feed video\.",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\([A-Za-z]+/[A-Za-z]+\)\s*$", "", text).strip()
    return get("title") or url, text


def explain_tfidf(vec, clf, pred, top_n):
    """Kata yang paling mendorong prediksi (khusus MultinomialNB)."""
    if not hasattr(clf, "feature_log_prob_"):
        return None
    classes = list(clf.classes_)
    i = classes.index(pred)
    lp = clf.feature_log_prob_
    contrib = vec[0] * (lp[i] - np.delete(lp, i, axis=0).mean(axis=0))
    idx = [j for j in np.argsort(contrib)[::-1][:top_n] if contrib[j] > 0]
    if not idx:
        return None
    names = tfidf.get_feature_names_out()
    return pd.DataFrame({"Kata": names[idx], "Kontribusi": contrib[idx]}).set_index(
        "Kata"
    )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Pengaturan")
    top_n = st.slider("Jumlah kata teratas ditampilkan", 5, 30, 10)
    st.divider()
    st.subheader("Info Model")
    st.write(f"Kosakata TF-IDF: **{len(tfidf.get_feature_names_out()):,}**")
    st.write(f"Kosakata Word2Vec: **{len(w2v.wv):,}**")
    st.write(f"Dimensi vektor W2V: **{w2v.vector_size}**")
    st.write("Kelas: " + ", ".join(map(str, clf_tfidf.classes_)))


# ---------------------------------------------------------------- input
def process(raw, source, title=None):
    with st.status("Memproses artikel...", expanded=True) as status:
        st.write("Membersihkan teks")
        st.write("Menghapus stopword & stemming")
        st.write("Mengubah teks menjadi vektor")
        st.write("Menjalankan klasifikasi")
        res = run_pipeline(raw)
        status.update(label="Selesai", state="complete", expanded=False)
    res["source"], res["title"] = source, title
    st.session_state["result"] = res


tab_url, tab_text = st.tabs(["Dari URL", "Dari Teks"])

with tab_url:
    url = st.text_input("Masukkan link artikel", placeholder="https://...")
    if st.button("Klasifikasi", key="btn_url", type="primary"):
        if not url.strip():
            st.error("URL tidak boleh kosong")
        else:
            try:
                with st.spinner("Mengambil artikel..."):
                    title, text = extract_url(url.strip())
                if not text:
                    st.error("Gagal mengekstrak artikel")
                else:
                    process(text, "URL", title)
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

with tab_text:
    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""
    c1, _ = st.columns([1, 4])
    if c1.button("Isi contoh"):
        st.session_state["text_input"] = CONTOH
    text = st.text_area("Masukkan teks artikel", height=200, key="text_input")
    if st.button("Klasifikasi", key="btn_text", type="primary"):
        if not text.strip():
            st.error("Teks tidak boleh kosong")
        else:
            process(text, "Teks")


# ---------------------------------------------------------------- hasil
res = st.session_state.get("result")
if res:
    st.divider()
    if not res["tokens"]:
        st.warning(
            "Tidak ada token tersisa setelah preprocessing. Coba teks yang lebih panjang."
        )
        st.stop()

    # ringkasan cepat
    m1, m2, m3 = st.columns(3)
    m1.metric("TF-IDF", res["pred_tfidf"])
    m2.metric("Skip-Gram", res["pred_w2v"])
    m3.metric("Jumlah token", len(res["tokens"]))
    if res["pred_tfidf"] == res["pred_w2v"]:
        st.success(f"Kedua model sepakat: **{res['pred_tfidf']}**")
    else:
        st.warning("Kedua model berbeda pendapat — lihat detail di tahap 5.")

    st.subheader("Bedah proses step by step")
    t1, t2, t3, t4, t5 = st.tabs(
        [
            "1. Input",
            "2. Cleaning",
            "3. Stopword & Stemming",
            "4. Vektorisasi",
            "5. Prediksi",
        ]
    )

    # 1. Input
    with t1:
        if res.get("title"):
            st.markdown(f"**Judul:** {res['title']}")
        st.caption(
            f"Sumber: {res['source']} • {len(res['raw'].split())} kata • {len(res['raw'])} karakter"
        )
        st.text_area("Teks asli", res["raw"], height=220, disabled=True)

    # 2. Cleaning
    with t2:
        st.caption("Tiga langkah berurutan; geser untuk melihat hasil tiap langkah.")
        langkah = st.select_slider(
            "Langkah",
            options=["Teks asli", "Lowercase", "Hapus URL", "Hapus angka & simbol"],
            value="Hapus angka & simbol",
        )
        key = {
            "Teks asli": "raw",
            "Lowercase": "lower",
            "Hapus URL": "no_url",
            "Hapus angka & simbol": "clean",
        }[langkah]
        st.text_area(
            "Hasil", re.sub(r"\s+", " ", res[key]).strip(), height=220, disabled=True
        )

    # 3. Stopword & stemming
    with t3:
        a, b, c = st.columns(3)
        a.metric("Kata sebelum", len(res["words"]))
        b.metric("Stopword/pendek dibuang", len(res["removed"]))
        c.metric("Token akhir", len(res["tokens"]))

        with st.expander("Kata yang dibuang"):
            rm = Counter(res["removed"]).most_common()
            st.dataframe(
                pd.DataFrame(rm, columns=["Kata", "Jumlah"]),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("**Hasil stemming**")
        hanya_berubah = st.checkbox("Tampilkan hanya yang berubah", value=True)
        rows = [
            {"Kata asli": k, "Hasil stem": v}
            for k, v in res["stem_map"].items()
            if not hanya_berubah or k != v
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown(f"**Token teratas (top {top_n})**")
        freq = pd.Series(Counter(res["tokens"]).most_common(top_n)).apply(pd.Series)
        freq.columns = ["Token", "Frekuensi"]
        st.bar_chart(freq.set_index("Token"))

    # 4. Vektorisasi
    with t4:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### TF-IDF")
            vec = res["vec_tfidf"][0]
            nz = np.nonzero(vec)[0]
            st.caption(f"{len(nz)} dari {len(vec):,} fitur bernilai > 0 (sparse).")
            names = tfidf.get_feature_names_out()
            top = nz[np.argsort(vec[nz])[::-1][:top_n]]
            st.bar_chart(pd.DataFrame({"Bobot": vec[top]}, index=names[top]))

        with col_b:
            st.markdown("#### Word2Vec Skip-Gram")
            uniq = set(res["tokens"])
            cov = len(set(res["in_vocab"])) / max(len(uniq), 1)
            st.progress(
                cov, text=f"Cakupan kosakata: {cov:.0%} token unik ada di model"
            )
            if res["oov"]:
                with st.expander(f"Token di luar kosakata ({len(set(res['oov']))})"):
                    st.write(", ".join(sorted(set(res["oov"]))))
            st.caption("Vektor dokumen = rata-rata vektor kata. 10 dimensi pertama:")
            st.dataframe(
                pd.DataFrame(res["vec_w2v"][0][:10].reshape(1, -1)), hide_index=True
            )

            kandidat = sorted(set(res["in_vocab"]))
            if kandidat:
                pilih = st.selectbox("Cari kata yang mirip dengan:", kandidat)
                sim = w2v.wv.most_similar(pilih, topn=5)
                st.dataframe(
                    pd.DataFrame(sim, columns=["Kata mirip", "Kemiripan"]),
                    hide_index=True,
                    use_container_width=True,
                )

    # 5. Prediksi
    with t5:
        col_a, col_b = st.columns(2)
        col_a.metric("TF-IDF + Naive Bayes", res["pred_tfidf"])
        col_b.metric("Skip-Gram + Naive Bayes", res["pred_w2v"])

        exp = explain_tfidf(res["vec_tfidf"], clf_tfidf, res["pred_tfidf"], top_n)
        if exp is not None:
            with st.expander(
                f"Kata yang paling mendorong prediksi TF-IDF ke '{res['pred_tfidf']}'"
            ):
                st.bar_chart(exp)
                st.caption(
                    "Kontribusi = bobot TF-IDF × selisih log-probabilitas kelas terprediksi vs rata-rata kelas lain."
                )
