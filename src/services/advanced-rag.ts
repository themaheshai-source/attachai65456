import type { RetrievedChunk } from '../domain/types.js';

export type EvidenceStatus = 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'UNCERTAIN' | 'NOT_SUPPORTED';
export type EvidenceDecision = {
  status: EvidenceStatus;
  coverage: number;
  count: number;
  sources: number;
  sufficient: boolean;
};
export type RetrievalDiagnostics = {
  query: string;
  expanded: boolean;
  profile: { documents: number; chunks: number; vocabulary: number; topTerms: string[] };
  candidateCounts: { exact: number; keyword: number; bm25: number; dense: number; fused: number };
  evidence: EvidenceDecision;
  rejected: Array<{ id: string; reason: string }>;
};
export type AdvancedRetrieval = {
  evidence: RetrievedChunk[];
  decision: EvidenceDecision;
  diagnostics: RetrievalDiagnostics;
};

export type RetrievalDiagnosticCallback = (stage: string, metrics: Record<string, unknown>) => void;

const STOPWORDS = new Set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'what', 'which', 'who', 'how', 'much', 'does', 'do', 'did', 'of', 'for', 'to', 'in', 'on', 'and', 'or', 'will', 'can', 'i', 'my', 'it', 'this', 'that', 'with', 'when', 'where', 'why']);
const SYNONYMS: Record<string, string> = {
  cancellation: 'cancel',
  canceled: 'cancel',
  cancelled: 'cancel',
  receive: 'refund',
  received: 'refund',
  back: 'refund',
  money: 'refund',
  flat: 'property',
  homes: 'property',
  home: 'property',
  prices: 'pricing',
  price: 'pricing',
  cost: 'pricing',
  costs: 'pricing',
  fee: 'pricing',
  fees: 'pricing',
  monthly: 'month',
  boxes: 'box',
  sizes: 'size',
  services: 'service',
};
const MAX_CANDIDATES = 40;
const RETRIEVAL_K = 20;
const FINAL_K = 8;
const CONTEXT_CHARS = 7000;

export const tokens = (value: string) =>
  (value.toLowerCase().match(/[\p{L}\p{N}_/-]+/gu) ?? [])
    .map((token) => SYNONYMS[token] ?? token)
    .filter((token) => !STOPWORDS.has(token));
const overlap = (query: Set<string>, text: string) => {
  const docTokens = new Set(tokens(text));
  return [...query].filter((token) => docTokens.has(token)).length;
};
const comparable = (text: string) => text.toLowerCase().replace(/\s+/g, ' ').trim();
const byScore = (items: Array<{ chunk: RetrievedChunk; score: number }>) =>
  items.sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id));

type SearchRankTrace = {
  rank: number;
  id: string;
  source: string;
  rankingScore: number | null;
  rankingScoreType: string;
  sourceChunkScore: number;
  matchType?: string;
  keywordHits?: number;
};
type SearchRankCallback = (trace: SearchRankTrace[]) => void;

export function exactSearch(query: string, chunks: RetrievedChunk[], limit = RETRIEVAL_K, onRank?: SearchRankCallback) {
  const normalized = comparable(query);
  const queryTerms = new Set(tokens(query));
  const phrases = chunks.filter((chunk) => normalized.length > 0 && comparable(chunk.text).includes(normalized));
  const phraseIds = new Set(phrases.map((chunk) => chunk.id));
  const restRanked = byScore(
    chunks
      .filter((chunk) => !phraseIds.has(chunk.id) && overlap(queryTerms, chunk.text) > 0)
      .map((chunk) => ({ chunk, score: overlap(queryTerms, chunk.text) })),
  );
  const results = [...phrases, ...restRanked.map((item) => item.chunk)].slice(0, limit);
  try {
    onRank?.(results.map((chunk, index) => {
      const phraseMatch = phraseIds.has(chunk.id);
      return {
        rank: index + 1,
        id: chunk.id,
        source: chunk.source,
        rankingScore: phraseMatch ? null : overlap(queryTerms, chunk.text),
        rankingScoreType: phraseMatch ? 'exact_phrase_match_priority_no_numeric_score' : 'query_token_overlap_count',
        sourceChunkScore: chunk.score,
        matchType: phraseMatch ? 'exact_phrase_match' : 'query_token_overlap',
      };
    }));
  } catch { /* Diagnostics do not alter retrieval behavior. */ }
  return results;
}

export function keywordSearch(
  query: string,
  chunks: RetrievedChunk[],
  limit = RETRIEVAL_K,
  onRank?: SearchRankCallback,
) {
  const queryTerms = new Set(tokens(query));

  const ranked = chunks
    .map((chunk) => ({
      chunk,
      hits: overlap(queryTerms, chunk.text),
    }))
    .filter((item) => item.hits > 0)
    .map((item) => ({
      chunk: item.chunk,
      score: item.hits + item.chunk.text.length / 1_000_000,
      hits: item.hits,
    }))
    .sort(
      (a, b) =>
        b.score - a.score ||
        a.chunk.id.localeCompare(b.chunk.id),
    )
    .slice(0, limit);

  try {
    onRank?.(
      ranked.map((item, index) => ({
        rank: index + 1,
        id: item.chunk.id,
        source: item.chunk.source,
        rankingScore: item.score,
        rankingScoreType:
          'keyword_hits_plus_length_tiebreaker',
        sourceChunkScore: item.chunk.score,
        matchType: 'keyword_overlap',
        keywordHits: item.hits,
      })),
    );
  } catch {
    /* Diagnostics must not alter retrieval behavior. */
  }

  return ranked.map((item) => item.chunk);
}

export function bm25Search(query: string, chunks: RetrievedChunk[], limit = RETRIEVAL_K, onRank?: SearchRankCallback) {
  const terms = tokens(query);
  const docs = chunks.map((chunk) => tokens(chunk.text));
  const averageLength = docs.reduce((sum, doc) => sum + doc.length, 0) / Math.max(docs.length, 1);
  const documentFrequency = new Map<string, number>();
  for (const doc of docs)
    for (const term of new Set(doc))
      documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
  const ranked = byScore(
    chunks
      .map((chunk, index) => {
        const doc = docs[index]!;
        const frequency = new Map<string, number>();
        for (const term of doc) frequency.set(term, (frequency.get(term) ?? 0) + 1);
        const score = terms.reduce((sum, term) => {
          const tf = frequency.get(term) ?? 0;
          if (!tf) return sum;
          const df = documentFrequency.get(term) ?? 0;
          const idf = Math.log(1 + (docs.length - df + 0.5) / (df + 0.5));
          return sum + (idf * tf * 2) / (tf + 1.5 * (0.25 + (0.75 * doc.length) / Math.max(averageLength, 1)));
        }, 0);
        return { chunk, score };
      })
      .filter((x) => x.score > 0),
  ).slice(0, limit);
  try {
    onRank?.(ranked.map((item, index) => ({ rank: index + 1, id: item.chunk.id, source: item.chunk.source, rankingScore: item.score, rankingScoreType: 'bm25_score', sourceChunkScore: item.chunk.score, matchType: 'bm25_term_match' })));
  } catch { /* Diagnostics do not alter retrieval behavior. */ }
  return ranked.map((item) => item.chunk);
}

export function evaluateEvidence(query: string, chunks: RetrievedChunk[]): EvidenceDecision {
  const queryTerms = new Set(tokens(query));
  const coverage = Math.max(
    0,
    ...chunks.map((chunk) => overlap(queryTerms, chunk.text) / Math.max(1, queryTerms.size)),
  );
  const exact =
    comparable(query).length > 0 &&
    chunks.some((chunk) => comparable(chunk.text).includes(comparable(query)));
  const strong = chunks.filter((chunk) => chunk.score >= 0.2).length;
  const topScore = Math.max(0, ...chunks.map((chunk) => chunk.score));
  const status: EvidenceStatus =
    exact || coverage >= 0.55 || strong >= 2 || topScore >= 0.5
      ? 'SUPPORTED'
      : (coverage >= 0.2 || topScore >= 0.2) && chunks.length
        ? 'PARTIALLY_SUPPORTED'
        : coverage > 0 && chunks.length
          ? 'UNCERTAIN'
          : 'NOT_SUPPORTED';
  return {
    status,
    coverage,
    count: chunks.length,
    sources: new Set(chunks.map((chunk) => chunk.source)).size,
    sufficient: status === 'SUPPORTED' || status === 'PARTIALLY_SUPPORTED',
  };
}

function profile(chunks: RetrievedChunk[]) {
  const frequencies = new Map<string, number>();
  for (const chunk of chunks)
    for (const term of tokens(chunk.text).filter((term) => term.length > 2))
      frequencies.set(term, (frequencies.get(term) ?? 0) + 1);
  return {
    documents: new Set(chunks.map((chunk) => chunk.source)).size,
    chunks: chunks.length,
    vocabulary: frequencies.size,
    topTerms: [...frequencies.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 50)
      .map(([term]) => term),
  };
}

function rrf(rankings: RetrievedChunk[][], limit = MAX_CANDIDATES) {
  const scores = new Map<string, number>();
  const found = new Map<string, RetrievedChunk>();
  for (const ranking of rankings)
    ranking.forEach((chunk, rank) => {
      const existing = found.get(chunk.id);
      if (!existing || chunk.score > existing.score) found.set(chunk.id, chunk);
      scores.set(chunk.id, (scores.get(chunk.id) ?? 0) + 1 / (60 + rank + 1));
    });
  return [...found.values()]
    .map((chunk) => ({ ...chunk, score: scores.get(chunk.id) ?? 0 }))
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id))
    .slice(0, limit);
}

function rerank(
  query: string,
  chunks: RetrievedChunk[],
  vocabulary: string[],
  denseScores?: Map<string, number>,
) {
  const queryTerms = new Set(tokens(query));
  const topTerms = new Set(vocabulary);
  const domainRatio =
    [...queryTerms].filter((term) => topTerms.has(term)).length / Math.max(1, queryTerms.size);
  return chunks
    .map((chunk) => {
      const lexical = overlap(queryTerms, chunk.text) / Math.max(1, queryTerms.size);
      const phrase = Number(
        comparable(query).length > 0 && comparable(chunk.text).includes(comparable(query)),
      );
      const domain = lexical > 0 ? domainRatio : 0;
      const lexicalScore = 0.65 * lexical + 0.25 * phrase + 0.1 * domain;
      const semanticScore = denseScores ? (denseScores.get(chunk.id) ?? 0) : chunk.score;
      const combinedScore =
        lexicalScore >= 0.999 || semanticScore >= 0.999
          ? 1
          : Math.max(semanticScore, lexicalScore);
      return { ...chunk, score: combinedScore };
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
}

function expandedQuery(query: string, chunks: RetrievedChunk[]) {
  const existing = new Set(tokens(query));
  const frequency = new Map<string, number>();
  for (const chunk of chunks)
    for (const term of tokens(chunk.text))
      if (!existing.has(term) && term.length > 2)
        frequency.set(term, (frequency.get(term) ?? 0) + 1);
  const additions = [...frequency.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, 5)
    .map(([term]) => term);
  return additions.length ? `${query} ${additions.join(' ')}` : query;
}

type SelectionDiagnostic = {
  id: string;
  source: string;
  score: number;
  textChars: number;
  rank: number;
  status: 'selected' | 'rejected' | 'skipped';
  reason: string;
};

function deduplicate(chunks: RetrievedChunk[], onDuplicate?: (chunk: RetrievedChunk, index: number, reason: string) => void) {
  const text = new Set<string>();
  return chunks.filter((chunk, index) => {
    const key = comparable(chunk.text);
    if (!key) { onDuplicate?.(chunk, index, 'empty_normalized_text'); return false; }
    if (text.has(key)) { onDuplicate?.(chunk, index, 'duplicate_normalized_text'); return false; }
    text.add(key);
    return true;
  });
}

function selectContext(chunks: RetrievedChunk[], onDecision?: (decision: SelectionDiagnostic) => void) {
  const selected: RetrievedChunk[] = [];
  // Avoid allocating diagnostic-only rank maps or decisions for ordinary requests.
  const originalRanks = onDecision ? new Map(chunks.map((chunk, index) => [chunk, index + 1])) : undefined;
  const deduplicated = deduplicate(chunks, onDecision ? (chunk, index, reason) => {
    onDecision({ id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, rank: index + 1, status: 'rejected', reason });
  } : undefined);
  let used = 0;
  for (let index = 0; index < deduplicated.length; index++) {
    const chunk = deduplicated[index]!;
    const rank = originalRanks?.get(chunk) ?? index + 1;
    if (chunk.score <= 0) {
      onDecision?.({ id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, rank, status: 'rejected', reason: 'non_positive_rerank_score' });
      continue;
    }
    const labelLength = `[Source ${selected.length + 1}: ${chunk.source} | chunk ${chunk.id}]\n`.length;
    if (used + labelLength + chunk.text.length > CONTEXT_CHARS) {
      onDecision?.({ id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, rank, status: 'rejected', reason: 'context_character_budget_exceeded' });
      for (const skipped of deduplicated.slice(index + 1)) {
        const skippedRank = originalRanks?.get(skipped) ?? index + 2;
        onDecision?.({ id: skipped.id, source: skipped.source, score: skipped.score, textChars: skipped.text.length, rank: skippedRank, status: 'skipped', reason: 'not_evaluated_after_context_budget_break' });
      }
      break;
    }
    selected.push(chunk);
    used += labelLength + chunk.text.length + 2;
    onDecision?.({ id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, rank, status: 'selected', reason: 'positive_score_and_within_context_character_budget' });
  }
  return selected;
}

export async function retrieveAdvancedEvidence(input: {
  question: string;
  corpus: RetrievedChunk[];
  dense: RetrievedChunk[];
  onDiagnostic?: RetrievalDiagnosticCallback;
}): Promise<AdvancedRetrieval> {
  const retrievalStarted = process.hrtime.bigint();
  const elapsedMs = (started: bigint) => Number(process.hrtime.bigint() - started) / 1_000_000;
  const diagnostic = (stage: string, metrics: Record<string, unknown>) => {
    try { input.onDiagnostic?.(stage, metrics); } catch { /* diagnostic callbacks must not alter retrieval */ }
  };
  const summarizeScores = (chunks: RetrievedChunk[]) => {
    const scores = chunks.map((chunk) => chunk.score).filter(Number.isFinite).sort((a, b) => a - b);
    return { min: scores[0] ?? null, median: scores.length ? scores[Math.floor((scores.length - 1) / 2)]! : null, max: scores.at(-1) ?? null };
  };
  const denseScores = new Map<string, number>();
  for (const chunk of input.dense) {
    denseScores.set(chunk.id, Math.max(denseScores.get(chunk.id) ?? 0, chunk.score));
  }
  const profileStarted = process.hrtime.bigint();
  const combinedCorpusCount = input.dense.length + input.corpus.length;
  const corpus = deduplicate([...input.dense, ...input.corpus]);
  const corpusProfile = profile(corpus);
  if (input.onDiagnostic) diagnostic('corpus_profile', { durationMs: elapsedMs(profileStarted), inputDenseCount: input.dense.length, inputCorpusCount: input.corpus.length, combinedCorpusCount, distinctCorpusCount: corpus.length, duplicateCount: combinedCorpusCount - corpus.length, documentCount: corpusProfile.documents, vocabularyCount: corpusProfile.vocabulary });
  const run = (query: string, dense: RetrievedChunk[]) => {
    const runStarted = process.hrtime.bigint();
    let phaseStarted = process.hrtime.bigint();
    let exactRankTrace: SearchRankTrace[] | undefined;
    const exact = exactSearch(query, corpus, RETRIEVAL_K, input.onDiagnostic ? (trace) => { exactRankTrace = trace; } : undefined);
    const exactDurationMs = elapsedMs(phaseStarted);
    phaseStarted = process.hrtime.bigint();
    let keywordRankTrace: SearchRankTrace[] | undefined;
    const keyword = keywordSearch(query, corpus, RETRIEVAL_K, input.onDiagnostic ? (trace) => { keywordRankTrace = trace; } : undefined);
    const keywordDurationMs = elapsedMs(phaseStarted);
    phaseStarted = process.hrtime.bigint();
    let bm25RankTrace: SearchRankTrace[] | undefined;
    const bm25 = bm25Search(query, corpus, RETRIEVAL_K, input.onDiagnostic ? (trace) => { bm25RankTrace = trace; } : undefined);
    const bm25DurationMs = elapsedMs(phaseStarted);
    phaseStarted = process.hrtime.bigint();
    const fused = rrf([exact, keyword, bm25, dense.slice(0, RETRIEVAL_K)]);
    const fusionDurationMs = elapsedMs(phaseStarted);
    phaseStarted = process.hrtime.bigint();
    const rerankedAll = rerank(query, fused, corpusProfile.topTerms, denseScores);
    const ranked = rerankedAll.slice(0, FINAL_K);
    const rerankedIds = new Set(rerankedAll.map((chunk) => chunk.id));
    const returnedIds = new Set(ranked.map((chunk) => chunk.id));
    const rerankRejected = fused.filter((chunk) => !rerankedIds.has(chunk.id)).map((chunk) => ({ id: chunk.id, source: chunk.source, score: chunk.score, reason: 'non_positive_rerank_score' }));
    const rankCutoffExcluded = rerankedAll.filter((chunk) => !returnedIds.has(chunk.id)).map((chunk, index) => ({ id: chunk.id, source: chunk.source, score: chunk.score, rank: FINAL_K + index + 1, reason: 'beyond_final_k_rank_limit' }));
    const rerankDurationMs = elapsedMs(phaseStarted);
    if (input.onDiagnostic) {
      const candidateRefs = (chunks: RetrievedChunk[], rankTrace?: SearchRankTrace[]) => chunks.slice(0, RETRIEVAL_K).map((chunk, index) => {
        const trace = rankTrace?.[index];
        return { rank: index + 1, id: chunk.id, source: chunk.source, score: trace ? trace.rankingScore : chunk.score, scoreType: trace?.rankingScoreType ?? 'provider_or_current_chunk_score', sourceChunkScore: chunk.score, ...(trace?.matchType ? { matchType: trace.matchType } : {}), ...(trace?.keywordHits === undefined ? {} : { keywordHits: trace.keywordHits }), textChars: chunk.text.length };
      });
      diagnostic('candidate_generation', {
        durationMs: elapsedMs(runStarted), exactDurationMs, keywordDurationMs, bm25DurationMs, fusionDurationMs, rerankDurationMs,
        query, queryChars: query.length, exactCount: exact.length, keywordCount: keyword.length, bm25Count: bm25.length, denseCount: dense.length, fusedCount: fused.length, rankedCount: ranked.length,
        exactScoreRange: summarizeScores(exact), keywordScoreRange: summarizeScores(keyword), bm25ScoreRange: summarizeScores(bm25), denseScoreRange: summarizeScores(dense), fusedScoreRange: summarizeScores(fused),
        exactCandidates: candidateRefs(exact, exactRankTrace), keywordCandidates: candidateRefs(keyword, keywordRankTrace), bm25Candidates: candidateRefs(bm25, bm25RankTrace), denseCandidates: candidateRefs(dense), fusedCandidates: candidateRefs(fused),
        rerankRejected, rankCutoffExcluded,
        ranked: ranked.map((chunk, index) => ({ rank: index + 1, id: chunk.id, source: chunk.source, score: chunk.score, scoreType: 'advanced_rag_rerank_score', textChars: chunk.text.length, text: chunk.text })),
      });
    }
    return { exact, keyword, bm25, fused, ranked };
  };
  const first = run(input.question, input.dense);
  let contextStarted = process.hrtime.bigint();
  let selectionDecisions: SelectionDiagnostic[] | undefined = input.onDiagnostic ? [] : undefined;
  let selected = selectContext(first.ranked, selectionDecisions ? (item) => selectionDecisions!.push(item) : undefined);
  let decision = evaluateEvidence(input.question, selected);
  const explainDecision = (query: string, chunks: RetrievedChunk[], currentDecision: EvidenceDecision) => {
    const queryComparable = comparable(query);
    const exactQueryMatch = queryComparable.length > 0 && chunks.some((chunk) => comparable(chunk.text).includes(queryComparable));
    const strongChunkCount = chunks.filter((chunk) => chunk.score >= 0.2).length;
    const topScore = Math.max(0, ...chunks.map((chunk) => chunk.score));
    const reason = exactQueryMatch ? 'exact_query_match'
      : currentDecision.coverage >= 0.55 ? 'query_coverage_at_least_0_55'
        : strongChunkCount >= 2 ? 'at_least_two_chunks_scored_0_2_or_higher'
          : topScore >= 0.5 ? 'top_chunk_score_at_least_0_5'
            : (currentDecision.coverage >= 0.2 || topScore >= 0.2) && chunks.length ? 'partial_support_threshold_met'
              : currentDecision.coverage > 0 && chunks.length ? 'nonzero_query_coverage_without_support_threshold'
                : 'no_evidence_met_support_thresholds';
    return { ...currentDecision, reason, exactQueryMatch, strongChunkCount, topScore, thresholds: { queryCoverageSupported: 0.55, strongChunkScore: 0.2, minimumStrongChunks: 2, topScoreSupported: 0.5, queryCoveragePartial: 0.2, topScorePartial: 0.2 } };
  };
  let decisionDiagnostics = input.onDiagnostic ? explainDecision(input.question, selected, decision) : undefined;
  if (input.onDiagnostic) diagnostic('context_selection', { durationMs: elapsedMs(contextStarted), query: input.question, candidateCount: first.ranked.length, selectedCount: selected.length, selectedChars: selected.reduce((sum, chunk) => sum + chunk.text.length, 0), contextCharacterLimit: CONTEXT_CHARS, selectionDecisions: selectionDecisions ?? [], selected: selected.map((chunk, index) => ({ rank: index + 1, id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, text: chunk.text })), decision, decisionDiagnostics });
  let expanded = false;
  let candidateCounts = {
    exact: first.exact.length,
    keyword: first.keyword.length,
    bm25: first.bm25.length,
    dense: input.dense.length,
    fused: first.fused.length,
  };
  const queryTerms = new Set(tokens(input.question));
  const seedChunks = first.fused.filter(
    (chunk) => overlap(queryTerms, chunk.text) > 0 || (denseScores.get(chunk.id) ?? 0) >= 0.2,
  );
  if (
    (decision.status === 'UNCERTAIN' || decision.status === 'NOT_SUPPORTED') &&
    seedChunks.length > 0
  ) {
    expanded = true;
    const expansionStarted = process.hrtime.bigint();
    const query = expandedQuery(input.question, seedChunks);
    if (input.onDiagnostic) diagnostic('query_expansion', { durationMs: elapsedMs(expansionStarted), executed: true, reason: 'initial_evidence_uncertain_or_not_supported_with_seed_candidates', query: input.question, expandedQuery: query, seedCount: seedChunks.length, inputQueryChars: input.question.length, expandedQueryChars: query.length, expandedByChars: Math.max(0, query.length - input.question.length) });
    const second = run(query, []);
    contextStarted = process.hrtime.bigint();
    selectionDecisions = input.onDiagnostic ? [] : undefined;
    selected = selectContext(
      rerank(
        input.question,
        rrf([first.fused, second.keyword, second.bm25]),
        corpusProfile.topTerms,
        denseScores,
      ).slice(0, FINAL_K),
      selectionDecisions ? (item) => selectionDecisions!.push(item) : undefined,
    );
    decision = evaluateEvidence(input.question, selected);
    decisionDiagnostics = input.onDiagnostic ? explainDecision(input.question, selected, decision) : undefined;
    if (input.onDiagnostic) diagnostic('context_selection_after_expansion', { durationMs: elapsedMs(contextStarted), query: input.question, candidateCount: second.fused.length, selectedCount: selected.length, selectedChars: selected.reduce((sum, chunk) => sum + chunk.text.length, 0), contextCharacterLimit: CONTEXT_CHARS, selectionDecisions: selectionDecisions ?? [], selected: selected.map((chunk, index) => ({ rank: index + 1, id: chunk.id, source: chunk.source, score: chunk.score, textChars: chunk.text.length, text: chunk.text })), decision, decisionDiagnostics });
    candidateCounts = {
      exact: first.exact.length + second.exact.length,
      keyword: first.keyword.length + second.keyword.length,
      bm25: first.bm25.length + second.bm25.length,
      dense: input.dense.length,
      fused: Math.min(MAX_CANDIDATES, first.fused.length + second.fused.length),
    };
  }
  if (!expanded && input.onDiagnostic) diagnostic('query_expansion', { durationMs: 0, executed: false, reason: 'initial_evidence_did_not_trigger_expansion', query: input.question, seedCount: seedChunks.length });
  const rejected = selected
    .filter((chunk) => chunk.score < 0.2)
    .map((chunk) => ({ id: chunk.id, reason: 'low deterministic rerank score' }));
  if (input.onDiagnostic) diagnostic('retrieval.completed', { durationMs: elapsedMs(retrievalStarted), evidenceCount: selected.length, candidateCounts, decision, expanded, rejectedCount: rejected.length, contextChars: selected.reduce((sum, chunk) => sum + chunk.text.length, 0) });
  return {
    evidence: selected,
    decision,
    diagnostics: {
      query: input.question,
      expanded,
      profile: corpusProfile,
      candidateCounts,
      evidence: decision,
      rejected,
    },
  };
}

/** Source-labelled, bounded whole-chunk context from the standalone baseline. */
export function buildEvidenceContext(chunks: RetrievedChunk[]) {
  return chunks
    .map((chunk, index) => `[Source ${index + 1}: ${chunk.source} | chunk ${chunk.id}]\n${chunk.text}`)
    .join('\n\n');
}

export function cleanAnswerScaffolding(answer: string) {
  const cleaned = answer
    .replace(/\\n/g, '\n')
    .replace(/^\s*(?:#{1,6}\s*)?(?:faq|answer)\s*:?\s*/i, '')
    .replace(/(?:^|\s)Q:\s*[^?]+\?\s*A:\s*/gi, ' ')
    .replace(/(?:^|\s)[QA]:\s*/gi, ' ')
    .replace(/\*{1,3}([^*\n]+)\*{1,3}/g, '$1')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
  const sentences = cleaned.match(/[^.!?]+[.!?]+|[^.!?]+$/g) ?? [];
  const seen = new Set<string>();
  return sentences
    .filter((sentence) => {
      const key = comparable(sentence);
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .join(' ')
    .trim();
}

// Compatibility alias retained for existing focused unit consumers. The active path uses retrieveAdvancedEvidence.
export function selectFocusedEvidence(
  candidates: RetrievedChunk[],
  question: string,
  maxChars = CONTEXT_CHARS,
) {
  const queryTerms = new Set(tokens(question));
  const denseScores = new Map(candidates.map((c) => [c.id, c.score]));
  const matching = candidates.filter((chunk) => overlap(queryTerms, chunk.text) > 0);
  const pool = matching.length ? matching : [];
  const ranked = rerank(
    question,
    rrf([
      exactSearch(question, pool),
      keywordSearch(question, pool),
      bm25Search(question, pool),
      pool,
    ]),
    profile(candidates).topTerms,
    denseScores,
  );
  const evidence: RetrievedChunk[] = [];
  let used = 0;
  for (const chunk of deduplicate(ranked)) {
    const faq = chunk.text.match(/(?:Q:\s*[^?]+\?\s*)?A:\s*(.*?)(?=\s*Q:|$)/i)?.[1]?.trim();
    const candidate = faq || chunk.text;
    if (used + candidate.length > maxChars) continue;
    evidence.push({ ...chunk, text: candidate });
    used += candidate.length;
    if (evidence.length >= 3) break;
  }
  return {
    evidence,
    rejected: candidates
      .filter((chunk) => !evidence.some((selected) => selected.id === chunk.id))
      .map((chunk) => ({ id: chunk.id, reason: 'not selected by bounded evidence selection' })),
  };
}
