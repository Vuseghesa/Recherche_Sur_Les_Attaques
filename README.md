# Recherche sur les attaques par empoisonnement

Attaque par empoisonnement : une menace réel sur les données ?

Objectif de notre étude : examiner le phénomène de l’attaque par empoisonnement de données et son impact sur la sécurité des méthodes d’apprentissage automatique.

Présentation du jeu de données.

Nous exploitons un ensemble de données (Cf. fichier : Iot_temp.csv ) contenant les relevés de température des dispositifs IoT installés à l'extérieur et à l'intérieur d'une pièce anonyme. Le dispositif était en phase de test. Il a donc été désinstallé ou éteint plusieurs fois pendant toute la période de prélèvement (du 28-07-2018 au 08-12-2018).
Comme détails techniques, ce jeu de données a 5 colonnes dont les étiquettes sont : id : identifiants uniques pour chaque relevé, room_id/id : identifiant de la pièce dans laquelle l'appareil a été installé (à l'intérieur et/ou à l'extérieur), noted_date : date et heure du relevé, temp : relevés de température et out/in : si le prélèvement a été effectué à partir d'un appareil installé à l'intérieur ou à l'extérieur de la pièce. Au total, ce jeu de données contient 97606 lignes. (Cf. https://www.openml.org/search?type=data&status=active&id=43351&sort=runs ). Cependant, pour répondre à certains tests d’analyse dont nous avions besoins, nous avons ajouté à ce jeu de données deux colonnes : inside et outside. Nous y avons également modifié certaines structures des données sans toucher au contenu afin de faciliter la lecture des résultats.

Description des codes

Dans cette approche, nous avons exploité les codes sources du chercheur KOHEI-MU (Cf. https://www.kaggle.com/code/koheimuramatsu/iot-temperature-forecasting ) mais en les adaptant à notre étude. La grande modification que nous y avons effectuée, c’est l’ajout d’un script qui permet l’empoisonnement de certains nœuds qui ont été sélectionnés aléatoirement. Ceci nous a permis de tester quatre types d’attaques  à savoir : l’attaque par modification des données, l’attaque par suppression de données, l’attaque par changement des labels et l’attaque d’empoisonnement par éponge.
En définitive, l’étude que nous avons effectuée sur ce jeu de données contenant les relevés des températures, nous a permis d’analyser et de comprendre l’impact des attaques par empoisonnement sur la sécurité des méthodes d’apprentissage automatique. Elle nous a permis également d’ouvrir notre perspective vers le développement des techniques de défense et de contre-mesures pour atténuer les différents risques liés à ces attaques devenues récurrentes ces derniers temps.
