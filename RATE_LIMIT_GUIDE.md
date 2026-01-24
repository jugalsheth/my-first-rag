# Rate Limit Fix Guide

## Why You're Hitting Rate Limits

**Free Tier Limits:**
- Gemini 2.5 Flash Lite: **10 requests/minute** (6 seconds between calls)
- Gemma 3 12B/4B: **30 requests/minute** (2 seconds between calls)
- Gemini 2.5 Flash: **5 requests/minute** (12 seconds between calls)

**Agentic RAG Usage:**
- Each iteration = **2-3 API calls** (generate answer, self-grade, refine query)
- 3 iterations max = **6-9 calls per question**
- Test suite (3 questions) = **18-27 calls total**

**The Math:**
- With 10 RPM limit: Can only make 10 calls per minute
- Test suite needs 18-27 calls → Takes 2-3 minutes minimum
- If you run too fast → **429 Rate Limit Error**

## What I Fixed

1. **Added delays to ALL API calls** (was missing on `generate_answer`)
2. **Increased minimum delay** from 6s to 7s (safer margin)
3. **Added delays between questions** in test suite (2x safety margin)
4. **Request tracking** to respect time windows
5. **Better rate limit messages** so you know what's happening

## Solutions

### Option 1: Wait It Out (Recommended for Testing)
```bash
# Just run the test suite - it now has automatic delays
python3 agentic_rag.py

# The system will:
# - Wait 7+ seconds between each API call
# - Wait 12+ seconds between questions
# - Show progress messages
```

### Option 2: Reduce Iterations
```bash
# Use fewer iterations to reduce API calls
python3 agentic_rag.py --iterations 1

# Or test one question at a time
python3 agentic_rag.py --single "Your question here"
```

### Option 3: Use Higher Tier Model
If you have access to Gemma models (30 RPM), they're faster:
- The system automatically tries Gemma first
- 30 RPM = 2 seconds between calls (much faster)

### Option 4: Upgrade API Tier
- Free tier: 5-30 RPM
- Paid tier: Much higher limits
- Check: https://ai.google.dev/pricing

### Option 5: Use Local Models (Advanced)
- Run models locally (Ollama, etc.)
- No rate limits
- Requires setup

## Current Behavior

**Before Fix:**
- ❌ No delay on first API call
- ❌ 6 second delays (too short)
- ❌ No delays between questions
- ❌ Hit rate limits immediately

**After Fix:**
- ✅ 7+ second delays between ALL calls
- ✅ 12+ second delays between questions
- ✅ Automatic rate limit respect
- ✅ Progress messages show delays

## Expected Timing

**Single Question (3 iterations):**
- 6-9 API calls × 7 seconds = **42-63 seconds**

**Test Suite (3 questions):**
- Question 1: ~60 seconds
- Delay: ~12 seconds
- Question 2: ~60 seconds
- Delay: ~12 seconds
- Question 3: ~60 seconds
- **Total: ~3-4 minutes**

## If You Still Hit Limits

1. **Wait 60 seconds** - Rate limits reset per minute
2. **Check your API key tier** - Free tier is very limited
3. **Reduce iterations**: `--iterations 1`
4. **Test one at a time**: `--single "question"`

## Monitoring

The system now shows:
- Which model you're using (and its RPM limit)
- Delay times between calls
- Progress messages during waits

Look for messages like:
```
✓ Using Gemini 2.5 Flash Lite (10 RPM)
Rate limit: 10 requests/minute = 6.0s delay between calls
Waiting 12.0s before next question to respect rate limits...
```
