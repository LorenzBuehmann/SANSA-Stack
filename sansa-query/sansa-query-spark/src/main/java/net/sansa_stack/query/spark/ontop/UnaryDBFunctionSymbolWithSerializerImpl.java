package net.sansa_stack.query.spark.ontop;

import it.unibz.inf.ontop.com.google.common.collect.ImmutableList;
import it.unibz.inf.ontop.model.term.functionsymbol.db.DBFunctionSymbolSerializer;
import it.unibz.inf.ontop.model.term.functionsymbol.db.impl.DBFunctionSymbolWithSerializerImpl;
import it.unibz.inf.ontop.model.type.DBTermType;

public class UnaryDBFunctionSymbolWithSerializerImpl extends DBFunctionSymbolWithSerializerImpl {

    protected UnaryDBFunctionSymbolWithSerializerImpl(String name, DBTermType inputDBType, DBTermType targetType,
                                                      boolean isAlwaysInjective,
                                                      DBFunctionSymbolSerializer serializer) {
        super(name, ImmutableList.of(inputDBType), targetType, isAlwaysInjective, serializer);
    }
}