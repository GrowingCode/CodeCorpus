package com.zzk;

import java.io.FileNotFoundException;

import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Phrase;
import com.itextpdf.text.pdf.PdfWriter;

public class CreateAddPhrase {
	public static void main(String args[]){
		Document document = new Document();		//创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\book.pdf"));// 关联对象与输出流
			document.open();// 打开文档	
			Phrase phrase1 = new Phrase("BeiJing Olympics");// 创建短语并添加内容
			document.add(phrase1);
			Phrase phrase2 = new Phrase("One world, one dream!");// 创建短语并添加内容
			document.add(phrase2);
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
	
}

