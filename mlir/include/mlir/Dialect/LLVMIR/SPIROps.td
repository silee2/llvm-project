//===-- SPIROps.td - SPIR IR dialect op definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the SPIR IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_SPIROPS_TD
#define MLIR_DIALECT_LLVMIR_SPIROPS_TD

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// SPIR dialect definitions
//===----------------------------------------------------------------------===//

def SPIR_Dialect : Dialect {
  let name = "spir";
  let cppNamespace = "::mlir::spir";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
  }];

  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// SPIR attribute definitions
//===----------------------------------------------------------------------===//

class SPIR_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SPIR_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}


//===----------------------------------------------------------------------===//
// SPIR op definitions
//===----------------------------------------------------------------------===//

class SPIR_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<SPIR_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// SPIR target attribute.
//===----------------------------------------------------------------------===//

def SPIR_TargettAttr :
    SPIR_Attr<"SPIRTarget", "target"> {
  let description = [{
    SPIR target attribute for controlling compilation of SPIR-V targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#spir.target] attributes {...} {
        ...
      }
    ```

    2. Target with `XeHPC` chip.
    ```
      gpu.module @mymodule [#spir.target<chip = "XeHPC">] {
        ...
      }
    ```
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"spirv64-unknown-unknown\"">:$triple,
    StringRefParameter<"Target chip.", "\"unknown\"">:$chip,
    StringRefParameter<"Target chip features.", "\"\"">:$features,
    OptionalParameter<"DictionaryAttr", "Target specific flags.">:$flags,
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link,
    OptionalParameter<"std::optional<spirv::VerCapExtAttr>", "SPIR-V vce triple.">:$vce
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features, $flags, $link, $vce)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"spirv64-unknown-unknown\"">:$triple,
                     CArg<"StringRef", "\"unknown\"">:$chip,
                     CArg<"StringRef", "\"\"">:$features,
                     CArg<"DictionaryAttr", "nullptr">:$targetFlags,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles,
                     CArg<"std::optional<spirv::VerCapExtAttr>", "std::nullopt">:$vceTriple), [{
      return Base::get($_ctxt, optLevel, triple, chip, features,
                       targetFlags, linkFiles, vceTriple);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool hasFlag(StringRef flag) const;
  }];
  let extraClassDefinition = [{
    bool $cppClass::hasFlag(StringRef flag) const {
      if (DictionaryAttr flags = getFlags())
        return flags.get(flag) != nullptr;
      return false;
    }
  }];
}

#endif // MLIR_DIALECT_LLVMIR_SPIROPS_TD
