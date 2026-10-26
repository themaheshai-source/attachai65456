import type { RetrievedChunk } from '../domain/types.js';

export type GroundingCheck = { ok: boolean; reasons: string[] };

type NumericClaim = { key: string; value: string; currency: string; qualifier: string };

const WORD_NUMBERS: Record<string, string> = {
  one: '1',
  two: '2',
  three: '3',
  four: '4',
  five: '5',
  six: '6',
  seven: '7',
  eight: '8',
  nine: '9',
  ten: '10',
};

/**
 * Normalize numeric claims for practical grounding. Currency is retained so a
 * supported pound amount cannot silently become an unsupported dollar amount;
 * omitted currency and harmless spacing/unit wording remain compatible.
 */
const numberTokens = (text: string, includeWordNumbers = false): NumericClaim[] => {
  const claims = [
    ...text.matchAll(
      /(£|AUD\s*\$|\$|€|₹)?\s*(\d+(?:[.,]\d+)?)(?:\s*%|\s*(?:per|a)\s*(?:month|year|day|week))?/gi,
    ),
  ].map((match) => {
    const raw = match[0].toLowerCase();
    const currency = (match[1] ?? '').replace(/\s+/g, '').toLowerCase();
    const value = match[2]!.replace(/,/g, '');
    const qualifier = /%/.test(raw)
      ? '%'
      : /\b(?:per|a)\s*(month|year|day|week)\b/i.exec(raw)?.[1]?.toLowerCase() ?? '';
    return { key: `${currency}:${value}:${qualifier}`, value, currency, qualifier };
  });
  if (includeWordNumbers) {
    for (const [word, digit] of Object.entries(WORD_NUMBERS)) {
      if (new RegExp(`\\b${word}\\b`, 'i').test(text)) {
        claims.push({ key: `:${digit}:`, value: digit, currency: '', qualifier: '' });
      }
    }
  }
  return claims;
};

const words = (text: string) =>
  new Set(
    (text.toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) ?? []).filter(
      (x) =>
        ![
          'what',
          'does',
          'have',
          'with',
          'from',
          'that',
          'this',
          'about',
          'are',
          'the',
          'how',
          'can',
          'you',
          'your',
          'for',
          'and',
        ].includes(x),
    ),
  );

function overlap(a: Set<string>, b: Set<string>) {
  let n = 0;
  for (const x of a) if (b.has(x)) n++;
  return n;
}

const UNNECESSARY_REFUSAL_RE =
  /\b(i don['’]?t have (?:verified |that )?information|sorry,?\s+i (?:have no|don['’]?t have) (?:that )?information|i do not know|unable to find (?:that|any) information)\b/i;

/**
 * Conservative, evidence-based checks for the most dangerous grounding failures.
 * It deliberately avoids trying to judge prose style or paraphrase quality.
 */
export function validateGrounding(
  question: string,
  answer: string,
  evidence: RetrievedChunk[],
): GroundingCheck {
  if (!evidence.length) return { ok: true, reasons: [] };
  const source = evidence.map((x) => x.text).join('\n');
  const reasons: string[] = [];
  const answerNumbers = numberTokens(answer, false);
  const sourceNumbers = numberTokens(source, true);
  const unsupported = answerNumbers.filter(
    (claim) =>
      !sourceNumbers.some(
        (supported) =>
          supported.value === claim.value &&
          (!claim.currency || !supported.currency || supported.currency === claim.currency) &&
          (!claim.qualifier || !supported.qualifier || supported.qualifier === claim.qualifier),
      ),
  );
  if (unsupported.length) reasons.push('unsupported numeric claim');

  const qWords = words(question);
  // A tenant/company name alone is not evidence that an UNKNOWN statement is
  // about the requested fact. Require two topical terms, excluding the
  // repeated brand token, to avoid unrelated UNKNOWN entries poisoning valid
  // answers from the same tenant document.
  const topicalQuestionWords = new Set([...qWords].filter((word) => !/^gousto$/i.test(word)));
  const sourceLines = source
    .split(/(?<=[.!?])\s+|\n/)
    .map((x) => x.trim())
    .filter(Boolean);
  const unknownLines = sourceLines.filter((x) =>
    /\bunknown\b|not available|not specified|cannot be verified/i.test(x),
  );
  const hasRelevantUnknown = unknownLines.some(
    (line) => overlap(topicalQuestionWords, words(line)) >= 2,
  );
  if (
    hasRelevantUnknown &&
    answerNumbers.length &&
    !/\b(unknown|not available|not specified|cannot verify|check the official|support)\b/i.test(
      answer,
    )
  ) {
    reasons.push('answer supplies a value where relevant evidence is unknown');
  }

  if (!hasRelevantUnknown && UNNECESSARY_REFUSAL_RE.test(answer)) {
    reasons.push('unnecessary refusal despite available tenant evidence');
  }

  const negativeRe = /\b(no|not|does not|do not|never|cannot|must not|without)\b/i;
  const negativeLines = sourceLines.filter((x) => negativeRe.test(x));
  const positiveLines = sourceLines.filter((x) => !negativeRe.test(x));
  const qualifiedLines = sourceLines.filter((x) =>
    /\b(most|some|vary|varies|depends|subject to|may|can differ)\b/i.test(x),
  );
  const affirmative =
    /\b(is|are|does|do|offers?|provides?|guarantees?|available|included|open|free)\b/i.test(
      answer,
    ) && !/\b(not|no|cannot|unavailable|unknown|should not)\b/i.test(answer);
  const aWords = words(answer);
  const minOverlap = Math.min(2, Math.max(1, topicalQuestionWords.size));
  const contradictsNegative =
    affirmative &&
    negativeLines.some((line) => {
      const lWords = words(line);
      const qHit = overlap(topicalQuestionWords, lWords) >= minOverlap;
      const aHit = overlap(aWords, lWords) >= 1;
      const supportedByPositive = positiveLines.some(
        (pos) => overlap(aWords, words(pos)) > overlap(aWords, lWords),
      );
      return qHit && aHit && !supportedByPositive;
    });
  if (contradictsNegative) reasons.push('possible contradiction with negative evidence');

  const answerHasQualification =
    /\b(most|some|vary|varies|depends|subject to|may|can differ|except|however|but)\b/i.test(
      answer,
    );
  if (
    /\b(all|every|always|never)\b/i.test(answer) &&
    !answerHasQualification &&
    qualifiedLines.some((line) => overlap(qWords, words(line)) >= 1)
  ) {
    reasons.push('answer drops an evidence qualification');
  }

  return { ok: reasons.length === 0, reasons };
}

export function groundingCorrection(reasons: string[]) {
  return `Grounding check failed (${reasons.join('; ')}). Re-read TENANT KNOWLEDGE as authoritative. Correct or remove every unsupported claim, number, price, availability statement, guarantee, or policy. If TENANT KNOWLEDGE contains the answer, answer directly from it instead of refusing. If the requested fact is absent or marked UNKNOWN, say it is not available in the current tenant knowledge. Return only the corrected customer-facing answer.`;
}
