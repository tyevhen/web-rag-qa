-- migrate:up
-- Keyword terms per language, used for both stored chunks and the question (db/repositories/documents.py).
-- Estonian (Postgres 16 has no estonian config): drop function words, then cut every word to its first 5 letters,
-- so inflected forms share a term (drooni, droonist -> droon). Stopwords: stopwords-iso/stopwords-et (MIT) plus
-- question words, pronouns, modal verbs and postpositions. Probe on golden_et: keyword-only gold in top 6 0.0 -> 0.84.
CREATE FUNCTION keyword_vector(lang text, content text) RETURNS tsvector
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT CASE lang
        WHEN 'et' THEN to_tsvector('simple', regexp_replace(
            regexp_replace(lower(content), '\m(aga|ajal|ei|et|ja|jah|jaoks|juba|ka|kas|kaua|kes|kohta|korral|kuhu|kui|kuidas|kus|kust|kõige|kõik|ma|me|meil|mida|midagi|miks|millal|milline|millised|milliseid|millist|mind|minu|mis|mitu|mu|mul|mulle|nad|need|neid|neil|nii|ning|ole|oled|olen|oli|oma|on|palju|peaks|peaksin|pean|pole|puhul|sa|saab|saan|seda|see|selle|siin|siis|sinu|sul|ta|tal|te|tee|teha|tuleb|tuleks|veel|väga|või|võib|võin|ära)\M', ' ', 'g'),
            '(\w{5})\w+', '\1', 'g'))
        ELSE to_tsvector('english', content)
    END
$$;
-- Term count of a keyword vector: the passage length BM25 normalises by
CREATE FUNCTION keyword_length(v tsvector) RETURNS int
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT coalesce(sum(cardinality(positions)), 0)::int FROM unnest(v)
$$;
-- Postgres 16 can't change a generated column's expression, so fts is rebuilt (its GIN index goes with it)
ALTER TABLE chunks DROP COLUMN fts;
ALTER TABLE chunks ADD COLUMN fts tsvector GENERATED ALWAYS AS (keyword_vector(metadata->>'lang', content)) STORED;
ALTER TABLE chunks ADD COLUMN fts_len int
    GENERATED ALWAYS AS (keyword_length(keyword_vector(metadata->>'lang', content))) STORED;
CREATE INDEX chunks_fts_gin_idx ON chunks USING gin (fts);

-- migrate:down
ALTER TABLE chunks DROP COLUMN fts_len;
ALTER TABLE chunks DROP COLUMN fts;
ALTER TABLE chunks ADD COLUMN fts tsvector GENERATED ALWAYS AS (
    to_tsvector(CASE metadata->>'lang' WHEN 'et' THEN 'simple'::regconfig ELSE 'english'::regconfig END, content)
) STORED;
CREATE INDEX chunks_fts_gin_idx ON chunks USING gin (fts);
DROP FUNCTION keyword_length(tsvector);
DROP FUNCTION keyword_vector(text, text);
