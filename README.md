# error-generation

Fully-controlled error generation for tabular data.

# Package API

**Input:**

| Name           | Age | Address          |
| -------------- | --- | ---------------- |
| John Doe       | 30  | Berlin, Germany  |
| Jane Doe       | 25  | Paris, France    |
| Mike Smith     | 40  | London, England  |
| Maria Garcia   | 35  | Madrid, Spain    |
| Ali Hassan     | 28  | Cairo, Egypt     |
| Anna Schmidt   | 32  | Munich, Germany  |
| Chen Wang      | 20  | Beijing, China   |
| Ayumi Sato     | 38  | Tokyo, Japan     |
| Ivan Petrov    | 45  | Moscow, Russia   |
| Sofia Martinez | 22  | Barcelona, Spain |

**Szenarien:**

1. ECAR: Füge mit 5% Wahrscheinlichkeit Tippfehler in Spalte "Name" ein. Ich möchte als Ausgabe
   1. die komplette Tabelle mit den fehlerhaften Werten und
   2. nur die fehlerhafte Spalte haben.
2. EAR: Wenn die Adresse in Deutschland, Frankreich oder China liegt, verdrehe Vor- und Nachnamen für 20% aller Zeilen deren Alter > 20 ist. Als Ausgabe möchte ich
   1. die komplette Tabelle mit den fehlerhaften Werten und
   2. nur die fehlerhafte Spalte haben.
3. EAR: Wenn 0 <= Age <= 25, dann wird die Adresse mit 10% permutiert, wenn 26 <= Age <= 30, dann mit 20%, sonst mit 30%. Als Ausgabe 1) und 2).
4. ENAR: Mir liegt eine weitere Spalte mit boolschen Werten namens "Geheimagent" vor. Wenn Werte in dieser Spalte TRUE sind, lösche Name. Gib 1) und 2) zurück.
5. ENAR: Ersetze Age mit Age/2 für alle Zeilen deren Age-Werte im obersten Quartil der Verteilung liegen. Gib 1) und 2) zurück.

---

## APIs

Wir unterscheiden in drei APIs: `high_level`, `mid_level` und `low_level`.

### High-Level API

Die high-level API soll bequem sein. Sie fügt irgendwelche Fehler in die Tabelle ein. Sie bietet deshalb wenig Einstellmöglichkeiten und bezieht sich auf die ganze Tabelle.

**Input**

| Argument             | Type            | Description                                                         |
| -------------------- | --------------- | ------------------------------------------------------------------- |
| table                | pd.DataFram     | The to-be-corrupted dataframe.                                      |
| error_rate           | float           | The fraction of cells to be erroneous.                              |
| preserve_dtypes=True | bool            | If true, only applies error types that preserve the table's dtypes. |
| mechanisms           | List[Mechanism] | _optional_ The mechanisms that shall be applied to insert errors.   |
| include_types        | List[ErrorType] | _optional_ The error types that shall be included.                  |
| exclude_types        | List[ErrorType] | _optional_ The error types that shall be excluded.                  |

**Output**
`Tuple[error_table: pd.DataFrame, error_mask: pd.DataFrame]`

- `error_table` is the resulting corrupted dataframe.
- `error_mask` is a dataframe containing boolean values that indicate error positions.

**Examples**

```python
from tab_err.api.high_level import create_errors
from tab_err.error.mechanism import ECAR, EAR, ENAR
from tab_err.error.type import Mojibake, Typo, Scale, WrongDType

data = ... # load data

# 1. don't care, just create whatever errors.
data_with_errors, error_mask = create_errors(
    data=data,
    # P: Ich schlage vor, wir nennen das Attribut "error_rate"
    # statt "rate" - ersteres ist expliziter.
    #rate=0.05
    error_rate=0.05  # 5% errors
)

# 1.1 I don't care, but _really_ mess up my data.
data_with_errors, error_mask = create_errors(
    data=data,
    error_rate=0.05,
    preserve_dyptes=False
)

# 2. some specific error mechanisms
data_with_errors, error_mask = create_errors(
    data=data,
    error_rate=0.05,
    mechanisms=[ECAR, ENAR]
)

# 3. specific error mechanism+type
data_with_errors, error_mask = create_errors(
    data=data,
    error_rate=0.05,
    mechanisms=[EAR],
    include_types=[Typo, Mojibake], # only these two
)

data_with_errors, error_mask = create_errors(
    data=data,
    error_rate=0.05,
    mechanisms=[EAR],
    exclude_types=[WrongDType], # everything but not this one
)
```

### Low-Level API

Die low-level API erlaubt genaue Definition der Fehlererstellung in einer Spalte. Sie regelt, wie viele Fehler welchen Typs mit welchem Mechanismus eingefügt werden.

**Input**

| Argument            | Type        | Description                                                                   |
| ------------------- | ----------- | ----------------------------------------------------------------------------- |
| table               | pd.DataFram | The to-be-corrupted dataframe.                                                |
| column              | str\|int    | The to-be-corrupted column.                                                   |
| error_rate          | float       | The fraction of cells to be erroneous.                                        |
| mechanism           | Mechanism   | The mechanism that shall be applied to insert errors.                         |
| error_type          | ErrorType   | The error types that shall be included.                                       |
| condition_to_column | str\|int    | _optional_ Column onto which the mechanism conditions the error distribution. |

- `condition_to_column` kann nötig sein, abhängig vom Fehlermechanismus.
- `low_level` API muss failen wenn Error Type und dtype der Spalte nicht kompatibel sind.

Fragen:

- Wie bezeichnen wir column? Über den Namen der Spalte mit `df.loc[:,column]`, oder über den Index mit `df.iloc[:,column]`? Ich fände es gut, wenn wir beides unterstützten. Ich habe regelmäßig mit Tabellen ohne Spaltennamen gearbeitet.

> P: Dieser Block passt aus meiner Sicht nicht zur low-level API: Dadurch, dass sich das Verhalten von `low_level.create_errors` verändert in Abhängigkeit der übergebenen Parametern, ist die API nicht mehr intuitiv. Sie würde überladen. Außerdem implizieren `include` und `exclude`, dass Standard `error_types` einer Spalte durch ihren dtype bestimmt werden. Das widerspricht der Absicht, dem Benutzer in der low-level API maximale Kontrolle zu geben.
>
> Ich schlage vor wir überlegen uns, wie wir das Verhalten in die mid-level API unterbekommen.
>
> ```
> - Tabelle, Column, Fehlerrate, Fehlermechanismus, optional: Depends on Column, optional: [include error types], optional: [exclude error types]
>     - Error Type wird zufällig anhand der möglichen gewählt
>     - berücksichtigt include/exclude parameter
> - Tabelle, Column, Fehlerrate, optional: Depends on Column, optional: [include error types], optional: [exclude error types]
>     - Error Type und Mechanismus is zufällig
> ```

**Output**
`Tuple[error_table: pd.DataFrame, error_mask: pd.DataFrame]`

- `error_table` is the resulting corrupted dataframe.
- `error_mask` is a DataFrame containing boolean values that indicate error positions.

**Examples**

```python
from tab_err.api.low_level import create_errors
from tab_err.error.mechanism import ECAR, EAR, ENAR
from tab_err.error.type import Mojibake, Typo, Scale, WrongDType

data = pd.read_parquet('./iris.parquet')

# 1. 5% random typos for the name column
data_with_errors, error_mask = create_errors(
    data=data,
    column="name",
    error_rate=0.05,
    mechanism=ECAR,
    type=Typo,
)
# 2. 5% mojibake in name column, conditioned on age
data_with_errors, error_mask = create_errors(
    data=data,
    column="name",
    error_rate=0.05,
    mechanism=EAR,
    type=Mojibake,
    # Ich finde das muss einen Fehler schmeißen in der low-level API wenn
    # conditioned_to_column nicht explizit gesetzt wurde.
    conditioned_to_column="age", # can be omitted to condition on random other column
)
# 3. 5% errors in name, mechanism and type will be random
# P: Das finde ich unintuitiv.
data_with_errors, error_mask = create_errors(
    data=data,
    column="name",
    error_rate=0.05,
    # include_types=[Typo, Mojibake],
    # exclude_types=[WrongDType],
)
```

### Mid-Level API

Die low-level API auf mehrere Spalten angewendet.
**Input Definitionen**

| Argument            | Type                 | Description                                                                   |
| ------------------- | -------------------- | ----------------------------------------------------------------------------- |
| table               | pd.DataFram          | The to-be-corrupted dataframe.                                                |
| columns             | List[str]\|List[int] | The to-be-corrupted columns.                                                  |
| error_rate          | float                | The fraction of cells to be erroneous.                                        |
| mechanisms          | List[Mechanism]      | _optional_ The mechanism that shall be applied to insert errors.              |
| error_type          | ErrorType            | _optional_ The error types that shall be included.                            |
| include_types       | List[ErrorType]      | _optional_ The error types that shall be included.                            |
| exclude_types       | List[ErrorType]      | _optional_ The error types that shall be excluded.                            |
| condition_to_column | str\|int             | _optional_ Column onto which the mechanism conditions the error distribution. |

- Fehler werden zufällig über columns verteilt anhand zufälliger Kombi der parameter.
- Desto mehr parameter weggelassen werden, desto ähnlicher wird es der high-level API

**Output**
`Tuple[error_table: pd.DataFrame, error_mask: pd.DataFrame]`

- `error_table` is the resulting corrupted dataframe.
- `error_mask` is a DataFrame containing boolean values that indicate error positions.

**Examples**

```python
from tab_err.api.mid_level import create_errors
from tab_err.error.mechanism import ECAR, EAR, ENAR
from tab_err.error.type import Mojibake, Typo, Scale, WrongDType

data = ... # load data

# equivalent to the high-level API
data_with_errors, error_mask = create_errors(
 data=data,
 columns=data.columns,
 rate=0.05,
)

# 1. 5% typos or mojibakes in firstname and lastname, conditioned to another column (EAR) or unknown information (ENAR)
data_with_errors, error_mask = create_errors(
 data=data,
 columns=["firstname", "lastname"],
 rate=0.05,
 mechanisms=[EAR, ENAR],
 include_types=[Typo, Mojibake],
)

# if all parameters are of length 1, it is equivalent to the low-level API
data_with_errors, error_mask = create_errors(
 data=data,
 columns=["age"],
 rate=0.05,
 mechanisms=[EAR],
 include_types=[Scale],
)
```

## Implementierung

### 1. Fehlermaske berechnen

- Boolsche Fehlermaske entspricht dem Erzeugen von Missing Values
- Die Fehlerrate is bereits definiert

### 2. Fehler einfügen

- einfach Fehler Typen, die passen einfügen?
  - oder gibt es hier nochmals die Möglichkeit in Abhängigkeit zu anderen Werten Fehler einzufügen?

**Probleme**

1. Wie kann man genaue Fehlerquote einstellen, wenn man diese über mehrere Spalten verteilt?
   - mehrfaches aufrufen der Low-Level API kann dazu führen, dass eine Zelle mehrfach ausgewählt wird => weniger Fehler als erwartet werden erzeugt
