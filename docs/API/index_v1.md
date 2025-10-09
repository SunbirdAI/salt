# SUNBIRDAI API

Welcome to the Sunbird AI API documentation. The Sunbird AI API provides you access to Sunbird's language models. The currently supported models are: 

- **Translation (English to Multiple)**: translate from English to Acholi, Ateso, Luganda, Lugbara and Runyankole.

- **Translation (Multiple to English)**: translate from the 5 local language above to English.

- **Speech To Text**: Convert speech audio to text. Currently the supported languages are (**English**, **Acholi**, **Ateso**, **Luganda**, **Lugbara** and **Runyankole**)


## Login and Signup
If you don't already have an account, visit the sunbird AI API page [here](https://api.sunbird.ai/). If You already have an account just proceed by logging in.

## Logging in and getting an access token.
Authentication is done via a Bearer token. After you have created an account and you are logged in just visit the [tokens](https://api.sunbird.ai/tokens) page to get your access token. This is the `auth token` that is required when making calls to the sunbird AI api.

To see the full api endpoint documentations, visit the api docs [here](https://api.sunbird.ai/docs).

## AI Tasks
- Use the `/tasks/stt` endpoint for speech to text inference for one audio file.
- Use the `tasks/nllb-translate` endpoint for translation of text input with the NLLB model.
- Use the `/tasks/language_id` endpoint for auto language detection of text input. 
This endpoint identifies the language of a given text. It supports a limited set 
of local languages including Acholi (ach), Ateso (teo), English (eng),Luganda (lug), 
Lugbara (lgg), and Runyankole (nyn).
- Use the `/tasks/summarise` endpoint for anonymised summarization of text input. 
This endpoint does anonymised summarisation of a given text. The text languages
supported for now are English (eng) and Luganda (lug).

## Getting started
The guides below demonstrate how to make endpoint calls to the api programmatically. Select your programming language of choice to see the example usage.

### Sunbird AI API Tutorial
This page describes how to use the Sunbird AI API and includes code samples in Python and Javascript.


### Part 1: How to authenticate
1. If you don't already have an account, create one at https://api.sunbird.ai/register and login.
2. Go to the [tokens page](https://api.sunbird.ai/tokens) to get your access token which you'll use to authenticate

Add an `.env` file in the same directory as the script and define `AUTH_TOKEN` in it:

```
AUTH_TOKEN=your_token_here
```

### Part 2: How to call the translation endpoint
Refer to the sample code below. Replace `{access_token}` with the token you received above.

=== "Python"

    Install the required dependencies:

    ```sh
    pip install requests python-dotenv
    ```

    ``` python
    import os
    import requests

    from dotenv import load_dotenv

    load_dotenv()

    url = "https://api.sunbird.ai/tasks/nllb_translate"
    access_token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    data = {
        "source_language": "lug",
        "target_language": "eng",
        "text": "Ekibiina ekiddukanya omuzannyo gw’emisinde mu ggwanga ekya Uganda Athletics Federation kivuddeyo nekitegeeza nga lawundi esooka eyemisinde egisunsulamu abaddusi abanakiika mu mpaka ezenjawulo ebweru w’eggwanga egya National Athletics Trials nga bwegisaziddwamu.",
    }

    response = requests.post(url, headers=headers, json=data)

    print(response.json())
    ```

=== "Javascript"

    Install the required dependencies:

    ```sh
    npm install axios dotenv
    ```

    ``` js
    const axios = require('axios');
    require('dotenv').config();

    const url = "https://api.sunbird.ai/tasks/nllb_translate";
    const accessToken = process.env.AUTH_TOKEN;

    const headers = {
        "accept": "application/json",
        "Authorization": `Bearer ${accessToken}`,
        "Content-Type": "application/json"
    };

    const data = {
        source_language: "lug",
        target_language: "eng",
        text: "Ekibiina ekiddukanya omuzannyo gw’emisinde mu ggwanga ekya Uganda Athletics Federation kivuddeyo nekitegeeza nga lawundi esooka eyemisinde egisunsulamu abaddusi abanakiika mu mpaka ezenjawulo ebweru w’eggwanga egya National Athletics Trials nga bwegisaziddwamu."
    };

    axios.post(url, data, { headers })
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.error(error.response ? error.response.data : error.message);
        });
    ```


The dictionary below represents the language codes available now for the translate endpoint

=== "Python"

    ``` python
    language_codes: {
        "English": "eng",
        "Luganda": "lug",
        "Runyankole": "nyn",
        "Acholi": "ach",
        "Ateso": "teo",
        "Lugbara": "lgg"
    }
    ```

=== "Javascript"

    ``` js
    const languageCodes = {
        English: "eng",
        Luganda: "lug",
        Runyankole: "nyn",
        Acholi: "ach",
        Ateso: "teo",
        Lugbara: "lgg"
    };
    ```


### Part 3: How to call the speech-to-text (ASR) endpoint
Refer to the sample code below. Replace `{access_token}` with the token you got from the `/auth/token` endpoint. And replace `/path/to/audio_file` with the path to the audio file you want to transcribe and `FILE_NAME` with audio filename. 

=== "Python"

    ``` python
    import os
    import requests

    from dotenv import load_dotenv

    load_dotenv()

    url = "https://api.sunbird.ai/tasks/stt"
    access_token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    files = {
        "audio": (
            "FILE_NAME",
            open("/path/to/audio_file", "rb"),
            "audio/mpeg",
        ),
    }
    data = {
        "language": "lug",
        "adapter": "lug",
        "whisper": True,
    }

    response = requests.post(url, headers=headers, files=files, data=data)

    print(response.json())
    ```

=== "Javascript"

    Install the required dependencies:

    ```sh
    npm install axios form-data dotenv
    ```

    ``` js
    const axios = require('axios');
    const FormData = require('form-data');
    const fs = require('fs');
    require('dotenv').config();

    const url = "https://api.sunbird.ai/tasks/stt";
    const accessToken = process.env.AUTH_TOKEN;

    const headers = {
        "accept": "application/json",
        "Authorization": `Bearer ${accessToken}`
    };

    // Create FormData
    const formData = new FormData();
    formData.append("audio", fs.createReadStream("/path/to/audio_file"), {
        filename: "FILE_NAME",
        contentType: "audio/mpeg"
    });
    formData.append("language", "lug");
    formData.append("adapter", "lug");
    formData.append("whisper", true);

    // Merge headers
    const requestHeaders = {
        ...headers,
        ...formData.getHeaders()
    };

    // Send POST request
    axios.post(url, formData, { headers: requestHeaders })
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.error(error.response ? error.response.data : error.message);
        });
    ```

### Part 4: How to call the summary endpoint

=== "Python"

    ``` python
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()

    url = "https://api.sunbird.ai/tasks/summarise"
    token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    text = (
        "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu "
        "eserbamby omwana oyo bingi bye yeegomba okuva mu buto bwe ate by'atasobola "
        "kwetuusaako bw'afuna mu naawumuwaamagezi nti ekya mazima nze kaboyiaadeyaatei "
        "ebintu kati bisusse mu uganda wano ebyegombebw'omwana by'atasobola kwetuusaako "
        "ng'ate abazadde nabo bambi bwe beetunulamubamufuna mpola tebasobola kulabirira "
        "mwana oyo bintu by'ayagala ekivaamu omwana akemererwan'ayagala omulenzi omulenzi "
        "naye n'atoba okuatejukira ba mbi ba tannategeera bigambo bya kufuna famire fulani "
        "bakola kyagenda layivu n'afuna embuto eky'amazima nze mbadde nsaba be kikwata "
        "govenment sembera embeera etuyisa nnyo abaana ne tubafaako embeera gwe nyiga gwa "
        "omuzadde olina olabirira maama we olina olabirira n'abato kati kano akasuumuseemu "
        "bwe ka kubulako ne keegulirayooba kapalaobakakioba tokyabisobola ne keyiiyabatuyambe "
        "buduufuembeera bagikyusa mu tulemye"
    )

    data = {"text": text}

    response = requests.post(url, headers=headers, json=data)

    print(response.json())
    ```

=== "Javascript"

    ``` js
    const axios = require('axios');
    require('dotenv').config();

    const url = "https://api.sunbird.ai/tasks/summarise";
    const token = process.env.AUTH_TOKEN;

    const headers = {
        "accept": "application/json",
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
    };

    const text = 
        "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu " +
        "eserbamby omwana oyo bingi bye yeegomba okuva mu buto bwe ate by'atasobola " +
        "kwetuusaako bw'afuna mu naawumuwaamagezi nti ekya mazima nze kaboyiaadeyaatei " +
        "ebintu kati bisusse mu uganda wano ebyegombebw'omwana by'atasobola kwetuusaako " +
        "ng'ate abazadde nabo bambi bwe beetunulamubamufuna mpola tebasobola kulabirira " +
        "mwana oyo bintu by'ayagala ekivaamu omwana akemererwan'ayagala omulenzi omulenzi " +
        "naye n'atoba okuatejukira ba mbi ba tannategeera bigambo bya kufuna famire fulani " +
        "bakola kyagenda layivu n'afuna embuto eky'amazima nze mbadde nsaba be kikwata " +
        "govenment sembera embeera etuyisa nnyo abaana ne tubafaako embeera gwe nyiga gwa " +
        "omuzadde olina olabirira maama we olina olabirira n'abato kati kano akasuumuseemu " +
        "bwe ka kubulako ne keegulirayooba kapalaobakakioba tokyabisobola ne keyiiyabatuyambe " +
        "buduufuembeera bagikyusa mu tulemye";

    const data = { text };

    axios.post(url, data, { headers })
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.error(error.response ? error.response.data : error.message);
        });
    ```

### Part 5: How to call the language_id endpoint

=== "Python"

    ``` python
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()

    url = "https://api.sunbird.ai/tasks/language_id"
    token = os.getenv("AUTH_TOKEN")
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    text = "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu"

    data = {"text": text}

    response = requests.post(url, headers=headers, json=data)

    print(response.json())
    ```

=== "Javascript"

    ``` js
    const axios = require('axios');
    require('dotenv').config();

    const url = "https://api.sunbird.ai/tasks/language_id";
    const token = process.env.AUTH_TOKEN;

    const headers = {
        "accept": "application/json",
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
    };

    const text = "ndowooza yange ku baana bano abato abatalina tufuna funa ya uganda butuufu";

    const data = { text };

    axios.post(url, data, { headers })
        .then(response => {
            console.log(response.data);
        })
        .catch(error => {
            console.error(error.response ? error.response.data : error.message);
        });
    ```

You can refer to the [docs](https://api.sunbird.ai/docs) for more info about the endpoints.

## Feedback and Questions.
Don't hesitate to leave us any feedback or questions you have by opening an [issue in this repo](https://github.com/SunbirdAI/sunbird-ai-api/issues).