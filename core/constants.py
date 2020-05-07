from core.imports.learn import RandomForestClassifier, KNeighborsClassifier, AdaBoostClassifier, SVC

DEFAULT_GRID_SEARCH_PARAMETERS = {type(RandomForestClassifier).__name__: {'n_estimators': [50, 100, 200],
                                                                          'class_weight': ['balanced', None],
                                                                          'min_samples_split': [0.2, 0.8, 2],
                                                                          'min_samples_leaf': [0.05, 0.5, 1],
                                                                          'max_features': ['auto', 'sqrt', None]}
                                  }
