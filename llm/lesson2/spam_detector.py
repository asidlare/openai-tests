# sms messages
'''
Sorry I missed your call let's talk when you have the time. I'm on 07090201529
HOT LIVE FANTASIES call now 08707509020 Just 20p per min NTT Ltd, PO Box 1327 Croydon CR9 5WB 0870 is a national rate call
You have received your mobile content. Enjoy
You have an important customer service announcement from PREMIER.
Thanks 4 your continued support Your question this week will enter u in2 our draw 4 £100 cash. Name the NEW US President? txt ans to 80082
Win the newest Harry Potter and the Order of the Phoenix (Book 5) reply HARRY, answer 5 questions - chance to be the first among readers!
FreeMSG You have been awarded a FREE mini DIGITAL CAMERA, just reply SNAP to collect your prize! (quizclub Opt out? Stop 80122300p/wk SP:RWM Ph:08704050406)
PRIVATE! Your 2003 Account Statement for 07753741225 shows 800 un-redeemed S. I. M. points. Call 08715203677 Identifier Code: 42478 Expires 24/10/04
U are subscribed to the best Mobile Content Service in the UK for £3 per ten days until you send STOP to 83435. Helpline 08706091795.
'''

from pydantic import BaseModel, Field, conlist
from typing import Literal

from llm.utils import get_response_with_instructor
import pprint


class SpamDetectorReasons(BaseModel):
    spam_reasons: conlist(str, max_length=3) = Field(
        ...,
        description="""
            A list of up to the 3 most important reasons explaining why the input is considered spam.
            Leave this list empty if no reasons are found.
        """
    )
    not_spam_reasons: conlist(str, max_length=3) = Field(
        ...,
        description="""
            A list of up to the 3 most important reasons explaining why the input is NOT considered spam.
            Leave this list empty if no reasons are found.
        """
    )
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="""
            A confidence score indicating the likelihood that the input is spam.
            A score of 100 means the input is 100% spam, while a score of 0% means it is 100% not spam.
        """
    )
    label: Literal["spam", "not_spam"]


def classify_text_message(text_message):

    messages = [
        {"role": "system", "content": "Classify text message as spam or not spam."},
        {"role": "user", "content": text_message}
    ]
    return get_response_with_instructor(
        messages=messages,
        response_model=SpamDetectorReasons,
        max_tokens=500
    )


if __name__ == "__main__":
    text_message = "U are subscribed to the best Mobile Content Service in the UK for £3 per ten days until you send STOP to 83435. Helpline 08706091795."
    response = classify_text_message(text_message)
    pprint.pprint(response.model_dump())
    print('*' * 50)

    test_message = "Sorry I missed your call let's talk when you have the time. I'm on 07090201529"
    response = classify_text_message(text_message)
    pprint.pprint(response.model_dump())
    print('*' * 50)

    text_message = "You have received your mobile content. Enjoy"
    response = classify_text_message(text_message)
    pprint.pprint(response.model_dump())
    print('*' * 50)

    test_message = "You have an important customer service announcement from PREMIER."
    response = classify_text_message(text_message)
    pprint.pprint(response.model_dump())
    print('*' * 50)
