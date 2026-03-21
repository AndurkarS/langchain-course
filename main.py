from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
def main():
    print("Hello from langchain-course!")
    information = """
    Ajit Kumar Doval (born 20 January 1945) is an Indian bureaucrat, spymaster and retired police and intelligence officer who has been serving as the longest tenured National Security Advisor of India (NSA) since 2014. Doval is serving his third consecutive five-year term as NSA. During his second tenure he was given Cabinet rank. Doval previously held the position of Director of the Intelligence Bureau from 2004 to 2005, after leading its operations wing for over a decade. He worked as a career intelligence officer for over 33 years. He is recognized for his contributions to counter-terrorism and covert missions.

    Born in Pauri Garhwal in present-day Uttarakhand, Doval completed his education from Ajmer Military School, Agra University, and the National Defence College. After clearing his Union Public Service Commission (UPSC) examination in 1968 he joined the Indian Police Service as a Kerala cadre officer. In 1972 he joined the Intelligence Bureau. His field assignments have spanned Mizoram, Sikkim, Punjab, and Jammu and Kashmir. He has held diplomatic assignments at the Indian High Commission in Islamabad and London. He received the Kirti Chakra gallantry award in 1989, becoming the first police officer to receive the second-highest peacetime military honour. Doval was also involved in multiple negotiations of hijacked Indian Airlines aircraft. At headquarters he was founder chairman of the Multi Agency Centre and the Joint Task Force on Intelligence.

    He retired as chief of the Intelligence Bureau in January 2005. In retirement he gave lectures, interviews and wrote op-eds. Around 2008, he helped set up the Rashtriya Raksha University. In 2009, he became founder director of the Vivekananda International Foundation, a public policy think tank based in New Delhi, and served as its director until his appointment as NSA. Major military operations during Doval's tenure include Operation Hot Pursuit, the Balakot airstrike and Operation Sindoor. The Doklam standoff was eventually resolved through diplomatic channels and negotiations.
    """

    summary_template = """
    Given information {information} about a person, I want you to do the following:
     1) A short summary
     2) 2 interesting facts about the person
    """

    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(model="gpt-5", temperature=0)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
