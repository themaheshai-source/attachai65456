import type { ClientConfig } from '../domain/types.js';

/** Detect equivalent English and common Roman-Hindi identity questions. */
export function isTenantIdentityQuestion(question: string): boolean {
  const text = question.trim().toLowerCase().replace(/[!?.,]+$/g, '').replace(/\s+/g, ' ');
  return /^(?:who are you|what are you|what(?:'s| is) your name|tell me about yourself|who created you|who developed you|what company do you belong to|which ai model are you|are you\s+\S+|(?:tum|tu|aap|ap)\s+(?:kaun|koun)\s+(?:ho|hain|hai)|(?:tumhara|aapka|apka)\s+naam\s+(?:kya|kia)\s+(?:hai|he))$/i.test(text);
}

/** A tenant configuration is authoritative for its customer-facing identity. */
export function tenantIdentityAnswer(question: string, config: ClientConfig): string | null {
  if (!isTenantIdentityQuestion(question)) return null;
  const assistant = config.assistantName.trim();
  const company = config.companyName?.trim();
  if (company) return `I'm ${company}'s AI assistant, ${assistant}. I'm here to help with ${company}'s products, services, and information.`;
  return `I'm ${assistant}. I'm here to help with this tenant's products, services, and information.`;
}
